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

"""Reward predictor model using diffusion features and Video_Former."""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from einops import rearrange

from cosmos_predict2._src.predict2.inference.diffusion_extract import Diffusion_feature_extractor
from cosmos_predict2._src.predict2.modules.video_former import Video_Former_3D


class RewardPredictor(nn.Module):
    """
    Reward predictor model that:
    1. Extracts features from clean video using diffusion model (no noise)
    2. Processes features through Video_Former
    3. Predicts reward via MLP
    """
    
    def __init__(
        self,
        feature_extractor: Diffusion_feature_extractor,
        feature_dim: int,
        video_former_dim: int = 512,
        video_former_depth: int = 4,
        video_former_heads: int = 8,
        video_former_num_latents: int = 64,
        video_former_num_frames: int = 16,
        num_reward_classes: int = 5,
        freeze_feature_extractor: bool = True,
        use_temporal: bool = False,
        module_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize reward predictor.
        
        Args:
            feature_extractor: Pre-trained diffusion feature extractor
            feature_dim: Dimension of extracted features (D from 5D features)
            video_former_dim: Dimension for Video_Former
            video_former_depth: Depth of Video_Former
            video_former_heads: Number of attention heads
            video_former_num_latents: Number of latent queries
            num_reward_classes: Number of reward classes (default: 5 for [0,1,2,3,4])
            freeze_feature_extractor: Whether to freeze feature extractor weights
            use_temporal: Whether to use temporal attention in Video_Former
        """
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        self.num_reward_classes = num_reward_classes
        self.video_former_num_latents = video_former_num_latents
        self.video_former_num_frames = video_former_num_frames
        self.compute_dtype = module_dtype or torch.bfloat16
        
        # Freeze feature extractor if requested
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()
        
        # Video_Former: processes 5D features (B, T, H, W, D) -> (B, num_latents, dim)
        # Input to Video_Former should be (B, T, H*W, D) -> (B, T, num_spatial_tokens, D)
        # We need to project feature_dim to video_former_dim first
        self.feature_proj = nn.Linear(feature_dim, video_former_dim, bias=True).to(dtype=self.compute_dtype)
        
        # Video_Former will handle variable frame lengths dynamically
        # num_frame is set to a reasonable default but will adapt to input
        self.video_former = Video_Former_3D(
            dim=video_former_dim,
            depth=video_former_depth,
            condition_dim=video_former_dim,  # After projection
            dim_head=video_former_dim // video_former_heads,
            heads=video_former_heads,
            num_latents=video_former_num_latents,
            num_frame=video_former_num_frames,
            num_time_embeds=max(video_former_num_frames, 64),
            ff_mult=4,
            activation='gelu',
            trainable=True,
            use_temporal=use_temporal,
        ).to(dtype=self.compute_dtype)
        
        # MLP for reward prediction
        # Input: (B, num_latents, video_former_dim)
        # Output: (B, num_reward_classes)
        self.reward_head = nn.Sequential(
            nn.LayerNorm(video_former_dim * video_former_num_latents),
            nn.Linear(video_former_dim * video_former_num_latents, video_former_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(video_former_dim * 2, video_former_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(video_former_dim, num_reward_classes),
        ).to(dtype=self.compute_dtype)
    
    def forward(
        self,
        video: torch.Tensor,
        text_prompt: Optional[Union[str, List[str]]] = None,
        timestep: float = 0.0,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            video: Input video tensor (B, C, T, H, W) or (C, T, H, W)
            text_prompt: Text prompt for conditioning (default: empty string)
            timestep: Timestep for diffusion (0.0 = no noise, clean video)
        
        Returns:
            reward_logits: (B, num_reward_classes)
        """
        # Extract features using diffusion model (no noise, timestep=0)
        # Returns: (B, T, H, W, D) where D is feature_dim
        if video.dim() == 4:
            video = video.unsqueeze(0)
        batch_size = video.shape[0]

        if text_prompt is None:
            prompt_list = ["a video"] * batch_size
        elif isinstance(text_prompt, str):
            prompt_list = [text_prompt] * batch_size
        else:
            prompt_list = list(text_prompt)
            if len(prompt_list) == 1 and batch_size > 1:
                prompt_list = prompt_list * batch_size
            elif len(prompt_list) != batch_size:
                raise ValueError(
                    f"Number of text prompts ({len(prompt_list)}) must match batch size ({batch_size})."
                )

        if text_embeddings is not None:
            text_embeddings = torch.as_tensor(text_embeddings, device=video.device)
            if text_embeddings.ndim == 2:
                text_embeddings = text_embeddings.unsqueeze(0)
            if text_embeddings.shape[0] != batch_size:
                raise ValueError(
                    f"Text embeddings batch dimension {text_embeddings.shape[0]} "
                    f"does not match video batch size {batch_size}."
                )

        with torch.set_grad_enabled(not self.feature_extractor.training):
            features_5d = self.feature_extractor(
                pixel_values=video,
                texts=prompt_list,
                timestep=timestep,
                all_layers=True,  # Concatenate all layer features
                text_embeddings=text_embeddings,
            )

        features_dtype = self.compute_dtype
        if features_5d.dtype != features_dtype:
            raise RuntimeError(
                f"Expected feature dtype {features_dtype}, but got {features_5d.dtype}. "
                "Ensure upstream autocast/data preparation matches the module precision."
            )

        B, T, H, W, D = features_5d.shape
        
        # Project features to video_former_dim
        # (B, T, H, W, D) -> (B, T, H, W, video_former_dim)
        features_5d_proj = self.feature_proj(features_5d)
        
        # Reshape to (B, T, H*W, video_former_dim) for Video_Former
        # Video_Former expects (B, T, n_features, d_visual)
        features_4d = rearrange(features_5d_proj, "b t h w d -> b t (h w) d")
        
        # Process through Video_Former
        # Input: (B, T, H*W, video_former_dim)
        # Output: (B, num_latents, video_former_dim)
        video_former_out = self.video_former(features_4d, mask=None)
        
        # Flatten for MLP
        # (B, num_latents, video_former_dim) -> (B, num_latents * video_former_dim)
        flattened = rearrange(video_former_out, "b n d -> b (n d)")
        
        # Predict reward
        reward_logits = self.reward_head(flattened)
        
        return reward_logits
    
    def predict_reward(
        self,
        video: torch.Tensor,
        text_prompt: Optional[Union[str, List[str]]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict reward class.
        
        Args:
            video: Input video tensor (B, C, T, H, W) or (C, T, H, W)
            text_prompt: Text prompt for conditioning
        
        Returns:
            reward_class: (B,) predicted reward class indices
        """
        with torch.no_grad():
            logits = self.forward(
                video,
                text_prompt=text_prompt,
                timestep=0.0,
                text_embeddings=text_embeddings,
            )
            return logits.argmax(dim=-1)

