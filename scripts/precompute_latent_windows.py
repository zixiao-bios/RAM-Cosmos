#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Precompute latent representations for video windows produced by RewardDataset.
# The script slices the dataset according to the specified window_size, encodes
# each window with the Cosmos tokenizer, and stores the resulting latents plus
# metadata to disk so that later training can load latents directly.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cosmos_predict2._src.predict2.datasets.reward_dataset import RewardDataset
from cosmos_predict2._src.predict2.inference.diffusion_extract import Diffusion_feature_extractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute latent video windows.")

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing videos.parquet and steps/.",
    )
    parser.add_argument(
        "--dataset-base",
        type=str,
        default=None,
        help="Base directory for video files. Defaults to dataset-dir.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Number of frames per video window.",
    )
    parser.add_argument(
        "--video-size",
        type=int,
        nargs=2,
        default=(240, 240),
        metavar=("HEIGHT", "WIDTH"),
        help="Target spatial size (H W) of decoded frames.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for latent encoding.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store latent files and manifest.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.csv",
        help="Filename for the manifest CSV stored in output-dir.",
    )
    parser.add_argument(
        "--latent-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Datatype used when saving latent tensors.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for Diffusion feature extractor (optional).",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Checkpoint path for Diffusion feature extractor (optional).",
    )
    parser.add_argument(
        "--s3-credential-path",
        type=str,
        default="",
        help="S3 credential path used by Video2WorldInference (optional).",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Context parallel size for the feature extractor.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1,
        help="Maximum number of video windows kept in in-memory cache by the dataset.",
    )
    parser.add_argument(
        "--no-lazy-steps",
        action="store_true",
        help="Disable lazy parquet loading and load the entire steps table into memory.",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default="head_rgb_video_path",
        help="Column name in videos.parquet for the video path.",
    )
    parser.add_argument(
        "--sample-cache-path",
        type=str,
        default=None,
        help="Optional path to JSON cache for the precomputed sample list.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for encoding (cuda or cpu).",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=0,
        help="Flush manifest to disk every N samples. 0 disables periodic flush.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards to process in parallel. "
        "Each run should use a unique shard-index in [0, num-shards).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of the shard to process (0-based).",
    )

    return parser.parse_args()


def normalize_video(videos: torch.Tensor) -> torch.Tensor:
    """Normalize videos to the [-1, 1] range expected by the tokenizer."""
    videos = videos.clone()
    max_val = videos.max()
    min_val = videos.min()

    if max_val > 1.1:
        videos = videos / 127.5 - 1.0
    elif 0.0 <= min_val and max_val <= 1.0:
        videos = videos * 2.0 - 1.0
    return videos


def ensure_list(values: Iterable) -> list:
    if isinstance(values, torch.Tensor):
        return values.cpu().tolist()
    return list(values)


def main() -> None:
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = output_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / args.manifest_name

    latent_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.latent_dtype]

    dataset_base = args.dataset_base or args.dataset_dir

    dataset = RewardDataset(
        dataset_dir=args.dataset_dir,
        dataset_base=dataset_base,
        window_size=args.window_size,
        video_size=tuple(args.video_size),
        cache_size=max(1, args.cache_size),
        video_key=args.video_key,
        shuffle=False,
        lazy_steps=not args.no_lazy_steps,
        sample_cache_path=args.sample_cache_path,
    )

    if args.num_shards > 1:
        shard_indices = [
            idx
            for idx in range(len(dataset))
            if idx % args.num_shards == args.shard_index
        ]
        if not shard_indices:
            raise ValueError(
                f"No samples assigned to shard {args.shard_index} "
                f"with num_shards={args.num_shards}."
            )
        dataset_subset = Subset(dataset, shard_indices)
        shard_info = (
            f"[Shard {args.shard_index + 1}/{args.num_shards}] "
            f"{len(shard_indices)} samples"
        )
    else:
        dataset_subset = dataset
        shard_indices = None
        shard_info = f"[Single shard] {len(dataset)} samples"

    dataloader = DataLoader(
        dataset_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    feature_extractor = Diffusion_feature_extractor(
        experiment_name=args.experiment_name,
        ckpt_path=args.ckpt_path,
        s3_credential_path=args.s3_credential_path,
        context_parallel_size=args.context_parallel_size,
    )
    tokenizer = feature_extractor.model.tokenizer
    tensor_kwargs = feature_extractor.model.tensor_kwargs

    manifest_fields = [
        "sample_index",  # local index within this run
        "global_index",  # original dataset index
        "latent_path",
        "reward",
        "video_id",
        "start_frame",
        "end_frame",
        "video_path",
        "video_shape",
        "latent_shape",
        "latent_dtype",
        "task_desc",
    ]

    sample_counter = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        writer.writeheader()

        with torch.no_grad():
            progress_desc = (
                f"Encoding latents {shard_info}"
                if args.num_shards > 1
                else "Encoding latents"
            )
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=progress_desc)):
                videos = batch["video"]  # (B, C, T, H, W)
                rewards = batch["reward"]
                video_ids = batch["video_id"]
                task_descs = batch["task_desc"]
                start_frames = ensure_list(batch["start_frame"])
                end_frames = ensure_list(batch["end_frame"])
                video_paths = batch["video_path"]

                if shard_indices is not None:
                    base_dataset_indices = [
                        shard_indices[batch_idx * args.batch_size + i]
                        for i in range(len(start_frames))
                    ]
                else:
                    base_dataset_indices = [
                        batch_idx * args.batch_size + i
                        for i in range(len(start_frames))
                    ]

                videos = videos.to(device, non_blocking=True)
                videos_norm = normalize_video(videos).to(**tensor_kwargs)

                latents = tokenizer.encode(videos_norm)
                latents = latents.to(latent_dtype).cpu()

                rewards_list = ensure_list(rewards)
                video_shape = list(videos.shape[1:])  # C, T, H, W

                for idx in range(latents.size(0)):
                    sample_counter += 1
                    latent_tensor = latents[idx]
                    reward_val = int(rewards_list[idx])
                    video_id = str(video_ids[idx])
                    start_frame = int(start_frames[idx])
                    end_frame = int(end_frames[idx])
                    video_path = str(video_paths[idx])
                    task_desc = str(task_descs[idx])
                    global_index = int(base_dataset_indices[idx])
                    latent_shape = list(latent_tensor.shape)

                    sample_uid = hashlib.md5(
                        f"{video_id}_{start_frame}_{end_frame}_{global_index}".encode()
                    ).hexdigest()
                    latent_filename = f"latent_{sample_uid}.pt"
                    latent_path = latents_dir / latent_filename

                    torch.save(
                        {
                            "latent": latent_tensor,
                            "reward": reward_val,
                            "video_id": video_id,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "video_path": video_path,
                            "task_desc": task_desc,
                            "video_shape": video_shape,
                            "latent_shape": latent_shape,
                            "window_size": args.window_size,
                            "global_index": global_index,
                        },
                        latent_path,
                    )

                    writer.writerow(
                        {
                            "sample_index": sample_counter,
                            "global_index": global_index,
                            "latent_path": str(latent_path.relative_to(output_dir)),
                            "reward": reward_val,
                            "video_id": video_id,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "video_path": video_path,
                            "video_shape": json.dumps(video_shape),
                            "latent_shape": json.dumps(latent_shape),
                            "latent_dtype": args.latent_dtype,
                            "task_desc": task_desc,
                        }
                    )

                    if args.save_frequency and sample_counter % args.save_frequency == 0:
                        manifest_file.flush()

                if device.type == "cuda":
                    torch.cuda.empty_cache()

    feature_extractor.cleanup()
    print(f"Saved {sample_counter} latent windows to {latents_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()

