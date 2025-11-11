# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset for loading ReWorld Dataset V0.1 with reward labels."""

import os
import pickle
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log

try:
    import av  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    av = None

try:
    import pyarrow.dataset as ds
except ImportError:  # pragma: no cover - optional dependency
    ds = None

import json


class RewardDataset(Dataset):
    """
    Dataset for loading ReWorld Dataset V0.1 with reward labels.
    
    Loads videos from parquet files and extracts fixed-length windows with corresponding rewards.
    Supports caching to speed up training.
    """
    
    def __init__(
        self,
        dataset_dir: str,
        dataset_base: str = "/inspire/hdd/project/robotsimulation/public/Dataset",
        window_size: int = 16,
        video_size: tuple[int, int] = (240, 240),
        cache_dir: Optional[str] = None,
        cache_size: int = 1000,
        video_key: str = "head_rgb_video_path",
        shuffle: bool = True,
        lazy_steps: bool = True,
        sample_cache_path: Optional[str] = None,
        text_embedding_cache: Optional[str] = None,
    ):
        """
        Initialize the reward dataset.
        
        Args:
            dataset_dir: Path to the dataset directory (containing videos.parquet and steps/)
            dataset_base: Base path for video files (videos.parquet paths are relative to this)
            window_size: Number of frames to extract per sample
            video_size: Target video size (H, W)
            cache_dir: Directory to cache loaded videos (None to disable caching)
            cache_size: Maximum number of videos to cache in memory
            video_key: Key in videos.parquet to use for video path
            shuffle: Whether to shuffle the dataset
        """
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.dataset_base = Path(dataset_base)
        self.window_size = window_size
        self.video_size = video_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_size = cache_size
        self.video_key = video_key
        self.shuffle = shuffle
        self.lazy_steps = lazy_steps
        self.sample_cache_path = Path(sample_cache_path) if sample_cache_path else None
        self.text_embedding_cache_path = Path(text_embedding_cache) if text_embedding_cache else None
        self.text_embedding_map: Optional[Dict[str, torch.Tensor]] = None
        
        # Load videos table
        videos_path = self.dataset_dir / "videos.parquet"
        if not videos_path.exists():
            raise FileNotFoundError(f"videos.parquet not found at {videos_path}")
        
        self.videos_df = pd.read_parquet(videos_path)
        log.info(f"Loaded {len(self.videos_df)} videos from {videos_path}")
        
        # Prepare steps loader
        steps_dir = self.dataset_dir / "steps"
        if not steps_dir.exists():
            raise FileNotFoundError(f"steps directory not found at {steps_dir}")

        if self.lazy_steps:
            if ds is None:
                raise ImportError(
                    "pyarrow is required for lazy step loading. "
                    "Install with `pip install pyarrow` or set lazy_steps=False."
                )
            self.steps_dataset = ds.dataset(str(steps_dir), format="parquet")
            log.info("Using lazy step loading backed by pyarrow.dataset")
            self.steps_df = None
        else:
            steps_files = sorted(steps_dir.glob("*.parquet"))
            if not steps_files:
                raise FileNotFoundError(f"No parquet files found in {steps_dir}")

            steps_dfs = [
                pd.read_parquet(f, columns=["video_id", "frame_idx", "reward"])
                for f in steps_files
            ]
            self.steps_df = pd.concat(steps_dfs, ignore_index=True)
            self.steps_df["video_id"] = self.steps_df["video_id"].astype("category")
            log.info(
                f"Loaded {len(self.steps_df)} steps from {len(steps_files)} files "
                "(eager mode)"
            )
            self.steps_dataset = None
        
        # Build sample list: (video_id, start_frame_idx, reward)
        if self.sample_cache_path and self.sample_cache_path.exists():
            try:
                with self.sample_cache_path.open("r", encoding="utf-8") as f:
                    cache_payload = json.load(f)

                if isinstance(cache_payload, dict) and "samples" in cache_payload:
                    cached_window_size = cache_payload.get("window_size")
                    if (
                        cached_window_size is not None
                        and cached_window_size != self.window_size
                    ):
                        log.info(
                            f"Sample cache window_size={cached_window_size} does not match current "
                            f"window_size={self.window_size}. Rebuilding cache."
                        )
                        raise ValueError("window_size mismatch")

                    self.samples = cache_payload["samples"]
                elif isinstance(cache_payload, list):
                    # Legacy cache format without metadata
                    self.samples = cache_payload
                else:
                    raise ValueError("Unrecognized cache format.")

                log.info(
                    f"Loaded {len(self.samples)} samples from cache {self.sample_cache_path}"
                )
            except Exception as exc:  # pragma: no cover - cache corruption
                log.warning(
                    f"Failed to load sample cache {self.sample_cache_path}: {exc}. "
                    "Rebuilding sample list."
                )
                self.samples = self._build_samples()
        else:
            self.samples = self._build_samples()

        log.info(f"Built {len(self.samples)} samples")

        if self.sample_cache_path and not self.sample_cache_path.exists():
            try:
                self.sample_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.sample_cache_path.open("w", encoding="utf-8") as f:
                    cache_payload = {
                        "window_size": self.window_size,
                        "video_size": list(self.video_size),
                        "num_samples": len(self.samples),
                        "samples": self.samples,
                    }
                    json.dump(cache_payload, f)
                log.info(f"Cached samples to {self.sample_cache_path}")
            except Exception as exc:  # pragma: no cover - cache write failure
                log.warning(f"Failed to write sample cache {self.sample_cache_path}: {exc}")

        self.text_embedding_map = None
        if self.text_embedding_cache_path:
            if not self.text_embedding_cache_path.exists():
                raise FileNotFoundError(
                    f"Text embedding cache not found at {self.text_embedding_cache_path}"
                )
            try:
                log.info(f"Loading text embeddings from {self.text_embedding_cache_path}")
                payload = torch.load(self.text_embedding_cache_path, map_location="cpu")
                if isinstance(payload, dict) and "embeddings" in payload:
                    embedding_map = payload["embeddings"]
                else:
                    embedding_map = payload
                if not isinstance(embedding_map, dict):
                    raise ValueError("Embedding cache must be a dict mapping prompt -> tensor.")
                converted_map: Dict[str, torch.Tensor] = {}
                for key, value in embedding_map.items():
                    tensor = torch.as_tensor(value)
                    if tensor.ndim == 3 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    if tensor.ndim != 2:
                        raise ValueError(
                            f"Embedding for prompt '{key}' has unexpected shape {tuple(tensor.shape)}; "
                            "expected (L, D)."
                        )
                    converted_map[str(key)] = tensor.contiguous()
                self.text_embedding_map = converted_map
                log.info(f"Loaded {len(self.text_embedding_map)} precomputed text embeddings")
            except Exception as exc:  # pragma: no cover - embedding cache corruption
                raise RuntimeError(
                    f"Failed to load text embedding cache {self.text_embedding_cache_path}: {exc}"
                ) from exc
        
        # Initialize cache
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_access_order: list = []
        
        # Setup preprocessing
        self.preprocess = T.Compose([
            T.ToTensor(),  # Convert to [0, 1] and (C, H, W)
        ])
    
    def _build_samples(self):
        """Build list of samples from videos and steps tables."""
        samples = []
        
        for _, video_row in self.videos_df.iterrows():
            video_id = video_row["video_id"]
            frame_num = video_row["frame_num"]
            video_path = video_row.get(self.video_key)
            
            if pd.isna(video_path):
                continue
            
            # Get steps for this video
            video_steps = self._get_steps_for_video(video_id)
            
            if len(video_steps) == 0:
                continue
            
            # Create samples: for each possible window start position
            max_start = max(0, frame_num - self.window_size)
            
            for start_idx in range(0, max_start + 1, self.window_size):
                end_idx = min(start_idx + self.window_size, frame_num)
                
                # Get reward for the last frame in the window (or average if multiple)
                window_steps = video_steps[
                    (video_steps["frame_idx"] >= start_idx) & 
                    (video_steps["frame_idx"] < end_idx)
                ]
                
                if len(window_steps) == 0:
                    continue
                
                # Use reward from the last frame in the window
                last_frame_idx = window_steps["frame_idx"].max()
                last_frame_step = window_steps[window_steps["frame_idx"] == last_frame_idx].iloc[0]
                reward = int(last_frame_step["reward"])
                
                samples.append({
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "start_frame": start_idx,
                    "end_frame": end_idx,
                    "reward": reward,
                    "task_desc": video_row.get("task_desc", ""),
                })
        
        if self.shuffle:
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(samples)
        
        return samples

    def _get_steps_for_video(self, video_id: str) -> pd.DataFrame:
        """Fetch step rows for a given video id."""
        if self.lazy_steps:
            table = self.steps_dataset.to_table(
                columns=["video_id", "frame_idx", "reward"],
                filter=ds.field("video_id") == video_id,
            )
            if table.num_rows == 0:
                return pd.DataFrame()
            df = table.to_pandas()
        else:
            df = self.steps_df[self.steps_df["video_id"] == video_id]

        return df.sort_values("frame_idx")
    
    def _get_cache_key(self, video_path: str, start_frame: int, end_frame: int) -> str:
        """Generate cache key for a video window."""
        key_str = f"{video_path}_{start_frame}_{end_frame}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_video_window(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load a window of frames from a video file."""
        full_path = self.dataset_base / video_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Video not found: {full_path}")
        
        # Check cache first
        cache_key = self._get_cache_key(video_path, start_frame, end_frame)
        if cache_key in self.cache:
            # Update access order
            if cache_key in self.cache_access_order:
                self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
            return self.cache[cache_key]
        
        if av is None:
            raise FileNotFoundError(
                f"Video not found: {full_path}"
            )

        try:
            video_tensor = self._load_video_window_pyav(
                full_path,
                start_frame,
                end_frame,
            )
            if video_tensor is None:
                raise RuntimeError(f"PyAV returned empty tensor for video {full_path}")
            self._put_video_in_cache(cache_key, video_tensor)
            return video_tensor
        except Exception as exc:  # pragma: no cover - optional dependency
            log.error(f"PyAV failed to load video {full_path}: {exc}")
            log.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load video {full_path} with PyAV: {exc}") from exc

    def _load_video_window_pyav(
        self,
        full_path: Path,
        start_frame: int,
        end_frame: int,
    ) -> Optional[torch.Tensor]:
        """Fallback video loader using PyAV for formats unsupported by decord."""
        if av is None:
            return None

        container = av.open(str(full_path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        target_len = self.window_size
        frames: list[torch.Tensor] = []
        last_frame: Optional[torch.Tensor] = None

        try:
            for idx, frame in enumerate(container.decode(stream)):
                if idx < start_frame:
                    continue
                if idx >= start_frame + target_len:
                    break

                img = frame.to_rgb().to_ndarray()
                frame_tensor = self.preprocess(img)
                frames.append(frame_tensor)
                last_frame = frame_tensor

            if not frames:
                return None

            while len(frames) < target_len:
                if last_frame is not None:
                    frames.append(last_frame.clone())
                else:
                    frames.append(torch.zeros_like(frames[0]))

            video_tensor = torch.stack(frames, dim=0)

            if video_tensor.shape[2:] != self.video_size:
                video_tensor = T.functional.resize(
                    video_tensor,
                    self.video_size,
                    antialias=True,
                )

            video_tensor = video_tensor.permute(1, 0, 2, 3)
            return video_tensor
        finally:
            container.close()
    
    def _load_video_window_decord(
        self,
        full_path: Path,
        start_frame: int,
        end_frame: int,
    ) -> torch.Tensor:
        vr = VideoReader(str(full_path), ctx=cpu(0), num_threads=2)
        try:
            total_frames = len(vr)

            desired_end = start_frame + self.window_size
            if desired_end > total_frames:
                frame_ids = list(range(start_frame, total_frames))
                if total_frames > 0:
                    frame_ids.extend([total_frames - 1] * (desired_end - total_frames))
                else:
                    frame_ids.extend([0] * (desired_end - start_frame))
            else:
                frame_ids = list(range(start_frame, desired_end))

            frame_data = vr.get_batch(frame_ids).asnumpy()
        finally:
            vr.seek(0)
            del vr

        frames = []
        for frame in frame_data:
            frame_tensor = self.preprocess(frame)
            frames.append(frame_tensor)

        video_tensor = torch.stack(frames, dim=0)

        if video_tensor.shape[2:] != self.video_size:
            video_tensor = T.functional.resize(
                video_tensor,
                self.video_size,
                antialias=True,
            )

        video_tensor = video_tensor.permute(1, 0, 2, 3)
        return video_tensor
    
    def _save_to_cache(self, cache_key: str, video_tensor: torch.Tensor):
        """Save video tensor to disk cache."""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(video_tensor, f)
            except Exception as e:
                log.warning(f"Failed to save cache for {cache_key}: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """Load video tensor from disk cache."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    log.warning(f"Failed to load cache for {cache_key}: {e}")
        return None
    
    def _put_video_in_cache(self, cache_key: str, video_tensor: torch.Tensor) -> None:
        """Store a video tensor in memory (and optionally on disk)."""
        if self.cache_dir:
            self._save_to_cache(cache_key, video_tensor)

        if cache_key in self.cache_access_order:
            self.cache_access_order.remove(cache_key)

        if len(self.cache) >= self.cache_size and self.cache_access_order:
            lru_key = self.cache_access_order.pop(0)
            if lru_key in self.cache:
                del self.cache[lru_key]

        self.cache[cache_key] = video_tensor
        self.cache_access_order.append(cache_key)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        video_tensor = self._load_video_window(
            sample["video_path"],
            sample["start_frame"],
            sample["end_frame"]
        )
        
        sample_dict = {
            "video": video_tensor,  # (C, T, H, W)
            "reward": torch.tensor(sample["reward"], dtype=torch.long),
            "video_id": sample["video_id"],
            "task_desc": sample["task_desc"],
            "start_frame": sample["start_frame"],
            "end_frame": sample["end_frame"],
            "video_path": sample["video_path"],
        }
        if self.text_embedding_map is not None:
            embedding = self.text_embedding_map.get(sample["task_desc"])
            if embedding is not None:
                sample_dict["text_embedding"] = embedding
        return sample_dict

