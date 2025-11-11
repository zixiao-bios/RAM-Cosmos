#!/usr/bin/env python3
"""
Training script for reward predictor model.

This script trains a reward predictor that:
1. Extracts features from clean videos using Cosmos-Predict2.5 diffusion model
2. Processes features through Video_Former
3. Predicts reward classes using MLP
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )
except ImportError:  # pragma: no cover
    FSDP = None  # type: ignore
    FullStateDictConfig = None  # type: ignore
    StateDictType = None  # type: ignore

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cosmos_predict2._src.predict2.datasets.reward_dataset import RewardDataset
from cosmos_predict2._src.predict2.inference.diffusion_extract import Diffusion_feature_extractor
from cosmos_predict2._src.predict2.models.reward_predictor import RewardPredictor
from cosmos_predict2.config import SetupArguments
from cosmos_predict2._src.imaginaire.utils import log

DEFAULT_FEATURE_DIM = 57344
DEFAULT_VIDEO_FORMER_NUM_FRAMES = 38
DEFAULT_VIDEO_FORMER_NUM_LATENTS = 8550


def parse_args():
    parser = argparse.ArgumentParser(description="Train reward predictor model")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to ReWorld Dataset V0.1 directory (containing videos.parquet and steps/)",
    )
    parser.add_argument(
        "--dataset-base",
        type=str,
        default="/inspire/hdd/project/robotsimulation/public/Dataset",
        help="Base path for video files (videos.parquet paths are relative to this)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Number of frames per video window",
    )
    parser.add_argument(
        "--video-size",
        type=int,
        nargs=2,
        default=[240, 240],
        help="Target video size (H, W)",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default="head_rgb_video_path",
        help="Key in videos.parquet to use for video path",
    )
    parser.add_argument(
        "--sample-cache-path",
        type=str,
        default=None,
        help="Optional path to JSON cache storing precomputed sample list.",
    )
    parser.add_argument(
        "--text-embedding-cache",
        type=str,
        default=None,
        help="Optional path to precomputed text embeddings (.pt) to skip online text encoder.",
    )
    parser.add_argument(
        "--no-lazy-steps",
        action="store_true",
        help="Disable pyarrow lazy loading of steps parquet files.",
    )
    
    # Model arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for loading diffusion model (uses default if None)",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Checkpoint path for loading diffusion model (uses default if None)",
    )
    parser.add_argument(
        "--s3-credential-path",
        type=str,
        default="",
        help="Path to S3 credentials file",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=None,
        help="Feature dimension from diffusion model (auto-detected if None)",
    )
    parser.add_argument(
        "--video-former-dim",
        type=int,
        default=512,
        help="Video_Former dimension",
    )
    parser.add_argument(
        "--video-former-depth",
        type=int,
        default=4,
        help="Video_Former depth",
    )
    parser.add_argument(
        "--video-former-heads",
        type=int,
        default=8,
        help="Number of attention heads in Video_Former",
    )
    parser.add_argument(
        "--video-former-num-latents",
        type=int,
        default=None,
        help="Number of latent queries in Video_Former (auto-computed if omitted)",
    )
    parser.add_argument(
        "--video-former-num-frames",
        type=int,
        default=None,
        help="Number of temporal bins for Video_Former (auto-computed if omitted)",
    )
    parser.add_argument(
        "--num-reward-classes",
        type=int,
        default=5,
        help="Number of reward classes",
    )
    parser.add_argument(
        "--freeze-feature-extractor",
        action="store_true",
        help="Freeze diffusion feature extractor weights",
    )
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="Use temporal attention in Video_Former",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/reward_predictor",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/reward_predictor",
        help="Directory for tensorboard logs",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache loaded videos (None to disable)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1000,
        help="Maximum number of videos to cache in memory",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl" if torch.cuda.is_available() else "gloo",
        help="Distributed backend to use (e.g., nccl, gloo)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--use-fsdp",
        action="store_true",
        help="Wrap the model with Fully Sharded Data Parallel (requires distributed training).",
    )
    
    return parser.parse_args()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def gather_model_state_dict(model, use_fsdp: bool):
    if use_fsdp:
        if FSDP is None or StateDictType is None or FullStateDictConfig is None:
            raise RuntimeError("FSDP is not available but was requested.")
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def gather_optimizer_state_dict(model, optimizer, use_fsdp: bool):
    if use_fsdp:
        if FSDP is None:
            raise RuntimeError("FSDP is not available but was requested.")
        # rank0_only=True ensures only rank0 receives the gathered optimizer state.
        return FSDP.full_optim_state_dict(model, optimizer, rank0_only=is_main_process())
    return optimizer.state_dict()


def load_optimizer_state_dict(model, optimizer, state_dict, use_fsdp: bool):
    if not state_dict:
        return
    if use_fsdp:
        if FSDP is None:
            raise RuntimeError("FSDP is not available but was requested.")
        FSDP.load_optim_state_dict(model, optimizer, state_dict)
    else:
        optimizer.load_state_dict(state_dict)


def train_epoch(
    model: RewardPredictor,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    sampler: Optional[torch.utils.data.Sampler] = None,
    distributed: bool = False,
    disable_progress: bool = False,
    text_embeddings_required: bool = False,
    target_dtype: Optional[torch.dtype] = None,
):
    """Train for one epoch."""
    if sampler is not None:
        sampler.set_epoch(epoch)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=disable_progress)
    for batch_idx, batch in enumerate(pbar):
        video_dtype = target_dtype if target_dtype is not None else batch["video"].dtype
        video = batch["video"].to(device=device, dtype=video_dtype, non_blocking=True)
        reward = batch["reward"].to(device)  # (B,)
        task_desc = batch.get("task_desc", [""] * reward.size(0))
        if isinstance(task_desc, (list, tuple)):
            text_prompts = [str(desc) for desc in task_desc]
        else:
            text_prompts = [str(task_desc)] * reward.size(0)

        text_embeddings = batch.get("text_embedding")
        if isinstance(text_embeddings, torch.Tensor):
            emb_dtype = target_dtype if target_dtype is not None else text_embeddings.dtype
            text_embeddings = text_embeddings.to(device=device, dtype=emb_dtype, non_blocking=True)
        else:
            text_embeddings = None
        if text_embeddings_required and text_embeddings is None:
            raise RuntimeError(
                "Text embedding cache was provided, but batch is missing 'text_embedding'."
            )
        
        # Forward pass
        optimizer.zero_grad()
        
        amp_enabled = target_dtype is not None and device.type == "cuda"
        autocast_dtype = target_dtype if amp_enabled else None
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype if autocast_dtype is not None else torch.float32,
            enabled=amp_enabled,
        ):
            logits = model(
                video,
                text_prompt=text_prompts,
                timestep=0.0,
                text_embeddings=text_embeddings,
            )
        loss = criterion(logits.float(), reward)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        batch_size = reward.size(0)
        total_loss += loss.item() * batch_size
        pred = logits.argmax(dim=-1)
        correct += (pred == reward).sum().item()
        total += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%" if total > 0 else "0.00%",
        })

    metrics = torch.tensor(
        [total_loss, correct, total],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(metrics)
    total_loss, correct, total = metrics.tolist()
    
    avg_loss = (total_loss / total) if total > 0 else 0.0
    accuracy = (100 * correct / total) if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(
    model: RewardPredictor,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    distributed: bool = False,
    disable_progress: bool = False,
    text_embeddings_required: bool = False,
    target_dtype: Optional[torch.dtype] = None,
):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=disable_progress):
            video_dtype = target_dtype if target_dtype is not None else batch["video"].dtype
            video = batch["video"].to(device=device, dtype=video_dtype, non_blocking=True)
            reward = batch["reward"].to(device)
            task_desc = batch.get("task_desc", [""] * reward.size(0))
            if isinstance(task_desc, (list, tuple)):
                text_prompts = [str(desc) for desc in task_desc]
            else:
                text_prompts = [str(task_desc)] * reward.size(0)

            text_embeddings = batch.get("text_embedding")
            if isinstance(text_embeddings, torch.Tensor):
                emb_dtype = target_dtype if target_dtype is not None else text_embeddings.dtype
                text_embeddings = text_embeddings.to(device=device, dtype=emb_dtype, non_blocking=True)
            else:
                text_embeddings = None
            if text_embeddings_required and text_embeddings is None:
                raise RuntimeError(
                    "Text embedding cache was provided, but batch is missing 'text_embedding'."
                )

            amp_enabled = target_dtype is not None and device.type == "cuda"
            autocast_dtype = target_dtype if amp_enabled else None
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype if autocast_dtype is not None else torch.float32,
                enabled=amp_enabled,
            ):
                logits = model(
                    video,
                    text_prompt=text_prompts,
                    timestep=0.0,
                    text_embeddings=text_embeddings,
                )
            loss = criterion(logits.float(), reward)
            
            batch_size = reward.size(0)
            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=-1)
            correct += (pred == reward).sum().item()
            total += batch_size

    metrics = torch.tensor(
        [total_loss, correct, total],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(metrics)
    total_loss, correct, total = metrics.tolist()
    
    avg_loss = (total_loss / total) if total > 0 else 0.0
    accuracy = (100 * correct / total) if total > 0 else 0.0
    
    return avg_loss, accuracy


def main():
    args = parse_args()
    
    # Distributed setup
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank == -1 and env_local_rank != -1:
        args.local_rank = env_local_rank
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1
    
    if args.distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=args.dist_backend)
    else:
        device = torch.device(args.device)
    
    if is_main_process():
        log.info(f"Using device: {device}")
    
    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=log_dir) if is_main_process() else None
    
    # Load dataset
    if is_main_process():
        log.info("Loading dataset...")
    sample_cache_path = Path(args.sample_cache_path) if args.sample_cache_path else None
    dataset = RewardDataset(
        dataset_dir=args.dataset_dir,
        dataset_base=args.dataset_base,
        window_size=args.window_size,
        video_size=tuple(args.video_size),
        cache_dir=args.cache_dir,
        cache_size=args.cache_size,
        video_key=args.video_key,
        shuffle=True,
        sample_cache_path=sample_cache_path,
        lazy_steps=not args.no_lazy_steps,
        text_embedding_cache=args.text_embedding_cache,
    )
    effective_window_size = dataset.window_size

    text_embeddings_available = dataset.text_embedding_map is not None

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    if is_main_process():
        log.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        if args.distributed
        else None
    )
    val_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
        if args.distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        sampler=val_sampler,
    )
    
    # Load feature extractor
    if is_main_process():
        log.info("Loading diffusion feature extractor...")
    feature_extractor = Diffusion_feature_extractor(
        experiment_name=args.experiment_name,
        ckpt_path=args.ckpt_path,
        s3_credential_path=args.s3_credential_path,
        disable_text_encoder=text_embeddings_available,
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    target_dtype = torch.bfloat16 if device.type == "cuda" else None

    feature_dim = args.feature_dim if args.feature_dim is not None else DEFAULT_FEATURE_DIM
    video_former_num_frames = (
        args.video_former_num_frames if args.video_former_num_frames is not None else DEFAULT_VIDEO_FORMER_NUM_FRAMES
    )
    video_former_num_latents = (
        args.video_former_num_latents
        if args.video_former_num_latents is not None
        else DEFAULT_VIDEO_FORMER_NUM_LATENTS
    )

    if args.feature_dim is None and is_main_process():
        log.info(f"Using default feature dimension: {feature_dim}")
    if args.video_former_num_frames is None and is_main_process():
        log.info(f"Using default Video_Former num_frames: {video_former_num_frames}")
    if args.video_former_num_latents is None and is_main_process():
        log.info(f"Using default Video_Former num_latents: {video_former_num_latents}")

    if video_former_num_latents % video_former_num_frames != 0:
        raise ValueError(
            "video_former_num_latents must be divisible by video_former_num_frames."
        )

    use_fsdp = args.use_fsdp
    if use_fsdp:
        if FSDP is None:
            raise ImportError("torch.distributed.fsdp is not available but --use-fsdp was set.")
        if not args.distributed:
            raise ValueError("--use-fsdp requires distributed execution (WORLD_SIZE > 1).")
        if device.type != "cuda":
            raise ValueError("--use-fsdp currently requires CUDA devices.")

    model = RewardPredictor(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        video_former_dim=args.video_former_dim,
        video_former_depth=args.video_former_depth,
        video_former_heads=args.video_former_heads,
        video_former_num_latents=video_former_num_latents,
        video_former_num_frames=video_former_num_frames,
        num_reward_classes=args.num_reward_classes,
        freeze_feature_extractor=args.freeze_feature_extractor,
        use_temporal=args.use_temporal,
        module_dtype=target_dtype,
    )
    model = model.to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    resume_checkpoint = None
    if args.resume:
        if is_main_process():
            log.info(f"Resuming from checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        start_epoch = resume_checkpoint.get("epoch", -1) + 1
        best_val_acc = resume_checkpoint.get("best_val_acc", resume_checkpoint.get("val_acc", 0.0))
    
    if use_fsdp:
        fsdp_kwargs = {"use_orig_params": True}
        if device.type == "cuda":
            fsdp_kwargs["device_id"] = device.index
        model = FSDP(model, **fsdp_kwargs)
    elif args.distributed:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_checkpoint is not None:
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            load_optimizer_state_dict(model, optimizer, optimizer_state, use_fsdp)
        resume_checkpoint = None
    
    # Training loop
    if is_main_process():
        log.info("Starting training...")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            sampler=train_sampler,
            distributed=args.distributed,
            disable_progress=not is_main_process(),
            text_embeddings_required=text_embeddings_available,
            target_dtype=target_dtype,
        )
        
        # Validate
        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
            distributed=args.distributed,
            disable_progress=not is_main_process(),
            text_embeddings_required=text_embeddings_available,
            target_dtype=target_dtype,
        )
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
        
        if is_main_process():
            log.info(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
            )
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            model_state = gather_model_state_dict(model, use_fsdp)
            optimizer_state = gather_optimizer_state_dict(model, optimizer, use_fsdp)
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            if is_main_process():
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer_state,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )
                log.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_state = gather_model_state_dict(model, use_fsdp)
            optimizer_state = gather_optimizer_state_dict(model, optimizer, use_fsdp)
            if is_main_process():
                best_path = save_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer_state,
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc,
                    },
                    best_path,
                )
                log.info(f"Saved best model (val_acc={val_acc:.2f}%) to {best_path}")
    
    if writer is not None:
        writer.close()
    if is_main_process():
        log.info("Training completed!")

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

