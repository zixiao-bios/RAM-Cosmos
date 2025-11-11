#!/usr/bin/env python3
"""
Precompute text embeddings for all task descriptions in the RewardDataset.

This script loads the dataset metadata, extracts unique task prompts, runs them
through the Cosmos Reason1 text encoder (Qwen2.5-VL-7B), and stores the resulting
embeddings to disk for reuse during training or latent precomputation.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.reward_dataset import RewardDataset
from cosmos_predict2._src.predict2.text_encoders.text_encoder import (
    TextEncoder,
    TextEncoderConfig,
)
from cosmos_predict2._src.common.types.embedding_concat_strategy import (
    EmbeddingConcatStrategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute text embeddings for task prompts.")

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing videos.parquet and steps/",
    )
    parser.add_argument(
        "--dataset-base",
        type=str,
        default="/inspire/hdd/project/robotsimulation/public/Dataset",
        help="Base path for video files (videos.parquet paths are relative to this).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=150,
        help="Window size used when constructing the dataset (matches training).",
    )
    parser.add_argument(
        "--video-size",
        type=int,
        nargs=2,
        default=[240, 240],
        help="Target video size (H, W). Only used to match dataset initialization.",
    )
    parser.add_argument(
        "--sample-cache-path",
        type=str,
        default=None,
        help="Optional path to an existing sample cache (reward_samples.json).",
    )
    parser.add_argument(
        "--no-lazy-steps",
        action="store_true",
        help="Disable pyarrow lazy loading of steps parquet files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the precomputed embeddings (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the text encoder on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for encoding prompts.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Datatype to store embeddings on disk.",
    )
    parser.add_argument(
        "--embedding-strategy",
        type=str,
        default=str(EmbeddingConcatStrategy.FULL_CONCAT),
        choices=[str(v) for v in EmbeddingConcatStrategy],
        help="Embedding concat strategy to match the diffusion model (default: full_concat).",
    )

    return parser.parse_args()


def chunk_list(items: List[str], chunk_size: int):
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def main():
    args = parse_args()

    device = torch.device(args.device)
    target_dtype = getattr(torch, args.dtype)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Initializing RewardDataset to collect task prompts...")
    dataset = RewardDataset(
        dataset_dir=args.dataset_dir,
        dataset_base=args.dataset_base,
        window_size=args.window_size,
        video_size=tuple(args.video_size),
        shuffle=False,
        sample_cache_path=args.sample_cache_path,
        lazy_steps=not args.no_lazy_steps,
    )

    task_prompts: Dict[str, int] = {}
    for sample in dataset.samples:
        prompt = str(sample.get("task_desc", "") or "")
        task_prompts[prompt] = task_prompts.get(prompt, 0) + 1

    unique_prompts = sorted(task_prompts.keys())
    log.info(f"Found {len(unique_prompts)} unique prompts (from {len(dataset.samples)} samples).")

    if not unique_prompts:
        log.warning("No prompts found in dataset; exiting without writing embeddings.")
        return

    log.info("Loading Cosmos Reason1 text encoder (Qwen2.5-VL-7B-Instruct)...")
    encoder_config = TextEncoderConfig(
        embedding_concat_strategy=args.embedding_strategy,
    )
    text_encoder = TextEncoder(encoder_config, device=str(device))
    text_encoder.model.eval()

    embeddings: Dict[str, torch.Tensor] = {}

    for batch_prompts in tqdm(list(chunk_list(unique_prompts, args.batch_size)), desc="Encoding prompts"):
        data_batch = {"ai_caption": batch_prompts}
        batch_embeddings = text_encoder.compute_text_embeddings_online(data_batch=data_batch, input_caption_key="ai_caption")

        if not isinstance(batch_embeddings, torch.Tensor):
            raise RuntimeError("Expected torch.Tensor from compute_text_embeddings_online.")

        batch_embeddings = batch_embeddings.to(dtype=target_dtype, device="cpu")

        for prompt, embedding in zip(batch_prompts, batch_embeddings, strict=True):
            embeddings[prompt] = embedding.clone()

    payload = {
        "embeddings": embeddings,
        "metadata": {
            "dtype": str(target_dtype),
            "num_prompts": len(embeddings),
            "prompt_frequencies": task_prompts,
        },
    }

    torch.save(payload, output_path)
    log.info(f"Saved embeddings for {len(embeddings)} prompts to {output_path}")


if __name__ == "__main__":
    main()


