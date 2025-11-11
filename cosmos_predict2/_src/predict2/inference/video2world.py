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

"""
# Script for generating I2W videos in s3
PYTHONPATH=. python cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root /project/cosmos/ybalaji/data/internal_val_set_clean

# Script for text2world generation
export EXPERIMENT=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/inference/video2world.py \
--experiment=${EXPERIMENT} \
--ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/${EXPERIMENT}/checkpoints/iter_000025000 \
--save_root results/base_model/${EXPERIMENT}_025k_seed0_t2w \
--num_latent_conditional_frames=0 --seed=0 \
--input_root /project/cosmos/fangyinw/data/pbench/v0

# I2W with context parallel with 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root /project/cosmos/ybalaji/data/internal_val_set_clean --context_parallel_size 8

# V2W with context parallel with 8 GPUs:
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_predict2/_src/predict2/inference/video2world.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4 --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22_qwen_concat_resume4/checkpoints/iter_000045000 --save_root results/cli_debug_from_s3 --input_root pbench_upsampled_prompts --num_latent_conditional_frames=2 --context_parallel_size=8


Folder structure:
We assume the input root contains images and prompts in the following format:
input_root/
 ├── image_1.jpg
 ├── image_1.txt
 ├── image_2.jpg
 └── image_2.txt
 └── ...

or videos and prompts in the following format:
input_root/
 ├── video_1.mp4
 ├── video_1.txt
 ├── video_2.mp4
 └── video_2.txt
 └── ...
"""

import argparse
import math
import os
from typing import List, Optional

from cosmos_predict2._src.imaginaire.flags import INTERNAL
import torch
import torchvision
from loguru import logger
from megatron.core import parallel_state
from PIL import Image

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
_VIDEO_EXTENSIONS = [".mp4"]
_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Video2World inference script."""
    parser = argparse.ArgumentParser(description="Image2World/Video2World inference script")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument("--num_video_frames", type=int, default=77, help="Number of video frames to generate")
    parser.add_argument("--guidance", type=int, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=1, help="Guidance value")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--resolution",
        type=str,
        default="none",
        help="Resolution of the video (H,W). Be default it will use model trained resolution. 9:16",
    )
    parser.add_argument("--input_root", type=str, default="assets/image2world", help="Input root")
    parser.add_argument("--save_root", type=str, default="results/image2world", help="Save root")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Custom negative prompt for classifier-free guidance. If not specified, uses default embeddings from S3.",
    )
    parser.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=1,
        help="Number of latent conditional frames (0, 1 or 2). For images, both values work by duplicating frames. For videos, uses the first N frames.",
    )
    # Context parallel arguments
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    parser.add_argument("--prompt_prefix", type=str, default="", help="Prompt prefix")
    return parser.parse_args()


def resize_input(video: torch.Tensor, resolution: list[int]):
    r"""
    Resizes and crops the input video tensor while preserving aspect ratio.

    The video is first resized so that the smaller dimension matches the target resolution,
    preserving the aspect ratio. Then, it's center-cropped to the target resolution.

    Args:
        video (torch.Tensor): Input video tensor of shape (T, C, H, W).
        resolution (list[int]): Target resolution [H, W].

    Returns:
        torch.Tensor: Resized and cropped video tensor of shape (T, C, target_H, target_W).
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, resolution)
    return video_cropped


def read_and_process_image(img_path: str, resolution: list[int], num_video_frames: int, resize: bool = True):
    """
    Reads an image, converts it to a video tensor, and processes it for model input.

    The image is loaded, converted to a tensor, and replicated to match the
    `num_video_frames`. It's then optionally resized and permuted to the
    standard video format (B, C, T, H, W).

    Args:
        img_path (str): Path to the input image file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): The number of frames the output video tensor should have.
        resize (bool, optional): Whether to resize the image to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W).

    Raises:
        ValueError: If the image extension is not one of the supported types.
    """
    ext = os.path.splitext(img_path)[1]
    if ext not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {ext}")

    # Read the image
    img = Image.open(img_path)

    # Convert to tensor
    img = torchvision.transforms.functional.to_tensor(img)
    # Create a video tensor by repeating the first frame
    vid_input = img.unsqueeze(0)  # Add temporal dimension T=1

    # Repeat the first frame to match the desired number of video frames
    # Note: The actual content for frames > 0 will be generated by the model.
    vid_input = torch.cat([vid_input, torch.zeros_like(vid_input).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
    vid_input = (vid_input * 255.0).to(torch.uint8)  # Convert to uint8 range if needed (might depend on model)
    if resize:
        # Resize and crop to the target resolution
        vid_input = resize_input(vid_input, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return vid_input


def read_and_process_video(
    video_path: str,
    resolution: list[int],
    num_video_frames: int,
    num_latent_conditional_frames: int = 2,
    resize: bool = True,
):
    """
    Reads a video, processes it for model input.

    The video is loaded using easy_io, and uses the last 4x(num_latent_conditional_frames - 1) + 1 from the video.
    If the video is shorter than num_video_frames, it pads with the last frame repeated.
    The first num_latent_conditional_frames are marked as conditioning frames.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        num_latent_conditional_frames (int): Number of latent conditional frames from the input video (1 or 2).
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T equals num_video_frames.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.

    Note:
        Uses the last 4x(num_latent_conditional_frames - 1) + 1 frames from the video. If video is shorter, pads with last frame repeated.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        logger.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    available_frames = video_tensor.shape[1]

    # Calculate how many frames to extract from input video
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
    logger.info(f"Will extract {frames_to_extract} frames from input video and pad to {num_video_frames}")

    # Validate num_latent_conditional_frames
    if num_latent_conditional_frames not in [1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 1 or 2, but got {num_latent_conditional_frames}")

    # Create output tensor with exact num_video_frames
    C, _, H, W = video_tensor.shape
    full_video = torch.zeros(C, num_video_frames, H, W)

    if available_frames < frames_to_extract:
        raise ValueError(
            f"Video has only {available_frames} frames but needs at least {frames_to_extract} frames for num_latent_conditional_frames={num_latent_conditional_frames}"
        )

    # Extract the last frames_to_extract from input video
    start_idx = available_frames - frames_to_extract
    extracted_frames = video_tensor[:, start_idx:, :, :]
    full_video[:, :frames_to_extract, :, :] = extracted_frames
    logger.info(f"Extracted last {frames_to_extract} frames from video (frames {start_idx} to {available_frames - 1})")

    # Pad remaining frames with the last extracted frame
    if frames_to_extract < num_video_frames:
        last_frame = extracted_frames[:, -1:, :, :]  # (C, 1, H, W)
        padding_frames = num_video_frames - frames_to_extract
        last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)  # (C, padding_frames, H, W)
        full_video[:, frames_to_extract:, :, :] = last_frame_repeated
        logger.info(f"Padded {padding_frames} frames with last extracted frame")

    # Convert to the format expected by the rest of the pipeline
    full_video = full_video.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
    full_video = (full_video * 255.0).to(torch.uint8)  # Convert to uint8 range

    if resize:
        # Resize and crop to the target resolution
        full_video = resize_input(full_video, resolution)

    # Convert to {B, C, T, H, W} format expected by the model
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Add batch dim B=1 and permute
    return full_video


class Video2WorldInference:
    """
    Handles the Video2World inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        context_parallel_size: int = 1,
        disable_text_encoder: bool = False,
    ):
        """
        Initializes the Video2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name (str): Name of the experiment configuration.
            ckpt_path (str): Path to the model checkpoint (local or S3).
            s3_credential_path (str): Path to S3 credentials file (if loading from S3).
            context_parallel_size (int): Number of GPUs for context parallelism.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.disable_text_encoder = disable_text_encoder
        self.process_group = None

        # Initialize distributed processing if context parallel size > 1
        if self.context_parallel_size > 1:
            self._init_distributed()

        # Load the model and config
        experiment_opts = []
        if not INTERNAL:
            experiment_opts.append("~data_train")
        if self.disable_text_encoder:
            experiment_opts.append("model.config.text_encoder_config.compute_online=false")
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
            load_ema_to_reg=True,
            experiment_opts=experiment_opts,
        )

        # Enable context parallel on the model if using context parallelism
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None

    def _init_distributed(self):
        """Initialize distributed processing for context parallelism."""

        # Initialize distributed environment
        distributed.init()

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        # Get the process group for context parallel
        self.process_group = parallel_state.get_context_parallel_group()

        logger.info(f"Initialized context parallel with size {self.context_parallel_size}")
        logger.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str | List[str],
        num_conditional_frames: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        use_neg_prompt: bool = True,
        precomputed_text_embeddings: Optional[torch.Tensor] = None,
        precomputed_neg_text_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            num_conditional_frames (int): Number of conditional frames to use.
            negative_prompt (str, optional): Custom negative prompt.
            use_neg_prompt (bool, optional): Whether to include negative prompt embeddings. Defaults to True.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape
        device = video.device

        if isinstance(prompt, str):
            prompts = [prompt] * B
        else:
            prompts = list(prompt)
            if len(prompts) == 1 and B > 1:
                prompts = prompts * B
            elif len(prompts) != B:
                raise ValueError(f"Expected {B} prompts, but got {len(prompts)}.")

        if use_neg_prompt:
            assert negative_prompt is not None, "Negative prompt is required when use_neg_prompt is True"
            if isinstance(negative_prompt, str):
                negative_prompts = [negative_prompt] * B
            else:
                negative_prompts = list(negative_prompt)
                if len(negative_prompts) != B:
                    raise ValueError(f"Expected {B} negative prompts, but got {len(negative_prompts)}.")
        else:
            negative_prompts = []

        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "fps": torch.randint(16, 32, (B,), device=device).float(),
            "padding_mask": torch.zeros(B, 1, H, W, device=device, dtype=video.dtype),
            "num_conditional_frames": num_conditional_frames,
        }

        if self.model.text_encoder is not None:
            data_batch["ai_caption"] = prompts
            if precomputed_text_embeddings is not None:
                emb = torch.as_tensor(precomputed_text_embeddings, device=video.device)
                if emb.ndim == 2:
                    emb = emb.unsqueeze(0)
                if emb.shape[0] != B:
                    raise ValueError(
                        f"Precomputed text embeddings batch dimension {emb.shape[0]} "
                        f"does not match video batch size {B}."
                    )
                data_batch["t5_text_embeddings"] = emb
                if use_neg_prompt:
                    if precomputed_neg_text_embeddings is None:
                        raise ValueError(
                            "Negative prompt embeddings are required when use_neg_prompt=True."
                        )
                    neg_emb = torch.as_tensor(precomputed_neg_text_embeddings, device=video.device)
                    if neg_emb.ndim == 2:
                        neg_emb = neg_emb.unsqueeze(0)
                    if neg_emb.shape[0] != B:
                        raise ValueError(
                            f"Precomputed negative embeddings batch dimension {neg_emb.shape[0]} "
                            f"does not match video batch size {B}."
                        )
                    data_batch["neg_t5_text_embeddings"] = neg_emb
            else:
                data_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": prompts, "images": None},
                    input_caption_key="ai_caption",
                )
                if use_neg_prompt:
                    data_batch["neg_t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                        data_batch={"ai_caption": negative_prompts, "images": None},
                        input_caption_key="ai_caption",
                    )
        else:
            if precomputed_text_embeddings is None:
                raise RuntimeError(
                    "Text encoder is disabled, but precomputed text embeddings were not provided."
                )
            emb = torch.as_tensor(precomputed_text_embeddings, device=video.device)
            if emb.ndim == 2:
                emb = emb.unsqueeze(0)
            if emb.shape[0] != B:
                raise ValueError(
                    f"Precomputed text embeddings batch dimension {emb.shape[0]} "
                    f"does not match video batch size {B}."
                )
            data_batch["t5_text_embeddings"] = emb
            if use_neg_prompt:
                if precomputed_neg_text_embeddings is None:
                    raise RuntimeError(
                        "Negative prompt embeddings are required when text encoder is disabled "
                        "and use_neg_prompt=True."
                    )
                neg_emb = torch.as_tensor(precomputed_neg_text_embeddings, device=video.device)
                if neg_emb.ndim == 2:
                    neg_emb = neg_emb.unsqueeze(0)
                if neg_emb.shape[0] != B:
                    raise ValueError(
                        f"Precomputed negative embeddings batch dimension {neg_emb.shape[0]} "
                        f"does not match video batch size {B}."
                    )
                data_batch["neg_t5_text_embeddings"] = neg_emb

        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.to(device="cuda").to(dtype=torch.bfloat16)

        return data_batch

    def generate_vid2world(
        self,
        prompt: str,
        input_path: str,
        guidance: int = 7,
        num_video_frames: int = 77,
        num_latent_conditional_frames: int = 1,
        resolution: str = "192,320",
        seed: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ):
        """
        Generates a video based on an input image or video and text prompt.

        Processes the input, prepares the data batch, runs the diffusion
        model sampling, and decodes the result into a video tensor.

        Args:
            prompt (str): The text prompt describing the desired video content/style.
            input_path (str): Path to the input image or video file.
            guidance (int, optional): Classifier-free guidance scale. Defaults to 7.
            num_video_frames (int, optional): Number of video frames to generate. Defaults to 77.
            num_latent_conditional_frames (int, optional): Number of latent conditional frames. Defaults to 1.
            resolution (str, optional): Target video resolution in "H,W" format. Defaults to "192,320".
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            negative_prompt (str, optional): Custom negative prompt. Defaults to the predefined default negative prompt.

        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
        """
        # Parse resolution string into tuple of integers
        if resolution == "none":
            h, w = self.model.get_video_height_width()
            video_resolution = (h, w)
        else:
            video_resolution = resolution.split(",")
            video_resolution = tuple([int(x) for x in video_resolution])
            assert len(video_resolution) == 2, "Resolution must be in 'H,W' format"

        # Get the correct number of frames needed by the model
        model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)

        # Determine if input is image or video and process accordingly
        if num_latent_conditional_frames > 0:
            ext = os.path.splitext(input_path)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                logger.info(f"Processing image input: {input_path}")
                vid_input = read_and_process_image(
                    img_path=input_path,
                    resolution=video_resolution,
                    num_video_frames=model_required_frames,
                    resize=True,
                )
            elif ext in _VIDEO_EXTENSIONS:
                logger.info(f"Processing video input: {input_path}")
                vid_input = read_and_process_video(
                    video_path=input_path,
                    resolution=video_resolution,
                    num_video_frames=model_required_frames,
                    num_latent_conditional_frames=num_latent_conditional_frames,
                    resize=True,
                )
            else:
                raise ValueError(
                    f"Unsupported file extension: {ext}. Supported extensions: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS}"
                )
        else:
            vid_input = torch.zeros(1, 3, model_required_frames, video_resolution[0], video_resolution[1]).to(
                torch.uint8
            )

        vid_input = vid_input[:, :3, :, :, :]

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            prompt,
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
        )

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"GPU memory usage after getting data_batch: {mem_bytes / (1024**3):.2f} GB")

        # Generate latent samples using the diffusion model
        # Video should be of shape torch.Size([1, 3, 93, 192, 320]) # Note: Shape check comment
        sample = self.model.generate_samples_from_batch(
            data_batch,
            n_sample=1,  # Generate one sample
            guidance=guidance,
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,  # Use classifier-free guidance
        )

        # Decode the latent sample into a video tensor
        video = self.model.decode(sample)

        return video

    def cleanup(self):
        """Clean up distributed resources."""
        if self.context_parallel_size > 1:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def main():
    torch.enable_grad(False)  # Disable gradient calculations for inference
    args = parse_arguments()

    # Validate num_latent_conditional_frames at the very beginning
    if args.num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(
            f"num_latent_conditional_frames must be 0, 1 or 2, but got {args.num_latent_conditional_frames}"
        )

    # Determine supported extensions based on num_latent_conditional_frames
    if args.num_latent_conditional_frames > 1:
        supported_extensions = _VIDEO_EXTENSIONS
        # Check if input folder contains any videos
        has_videos = False
        for file_name in os.listdir(args.input_root):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in _VIDEO_EXTENSIONS:
                has_videos = True
                break

        if not has_videos:
            raise ValueError(
                f"num_latent_conditional_frames={args.num_latent_conditional_frames} > 1 requires video inputs, "
                f"but no videos found in {args.input_root}. Found extensions: "
                f"{set(os.path.splitext(f)[1].lower() for f in os.listdir(args.input_root) if os.path.splitext(f)[1])}"
            )

        logger.info(f"Using video-only mode with {args.num_latent_conditional_frames} conditional frames")
    elif args.num_latent_conditional_frames == 1:
        supported_extensions = _IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS
        logger.info(f"Using image+video mode with {args.num_latent_conditional_frames} conditional frame")
    else:  # args.num_latent_conditional_frames == 0
        supported_extensions = _IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS
        logger.info(f"Using text-to-world mode with {args.num_latent_conditional_frames} conditional frames")

    # Initialize the inference handler with context parallel support
    s3_cred = ""

    video2world_cli = Video2WorldInference(
        args.experiment, args.ckpt_path, s3_cred, context_parallel_size=args.context_parallel_size
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Ensure save directory exists
    os.makedirs(args.save_root, exist_ok=True)

    # Process each file in the input directory
    for file_name in os.listdir(args.input_root):
        # Look for supported files
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext in supported_extensions:
            input_path = os.path.join(args.input_root, file_name)

            # Look for corresponding prompt file
            base_name = os.path.splitext(file_name)[0]
            prompt_path = os.path.join(args.input_root, base_name + ".txt")

            if not os.path.exists(prompt_path):
                logger.warning(f"No prompt file found for {file_name}, skipping...")
                continue

            with open(prompt_path, "r") as f:
                prompt = f.read().strip()

            if rank0:
                logger.info(f"Processing {file_name} with prompt: {prompt}")
                logger.info(f"Using {args.num_latent_conditional_frames} latent conditional frames")

            full_prompt = args.prompt_prefix + prompt
            video = video2world_cli.generate_vid2world(
                prompt=full_prompt,
                input_path=input_path,
                guidance=args.guidance,
                num_video_frames=args.num_video_frames,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                resolution=args.resolution,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
            )

            if rank0:
                output_name = os.path.splitext(file_name)[0]
                save_img_or_video((1.0 + video[0]) / 2, f"{args.save_root}/{output_name}", fps=16)
                logger.info(f"Saved video for {file_name} to {args.save_root}/{output_name}")
    # Synchronize all processes before cleanup
    if args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    video2world_cli.cleanup()


if __name__ == "__main__":
    main()
