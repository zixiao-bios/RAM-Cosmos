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
Feature extraction from Cosmos-Predict2.5 diffusion model.
Extracts intermediate features from each DiT block during a single denoising step.
"""

from typing import List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference, _DEFAULT_NEGATIVE_PROMPT
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.inference.utils import ensure_condition_channels
from cosmos_predict2.config import SetupArguments
from megatron.core import parallel_state


class Diffusion_feature_extractor(nn.Module):
    """
    Feature extractor for Cosmos-Predict2.5 diffusion model.
    
    Extracts intermediate features from each DiT block during a single denoising step.
    Similar to diffusion_extract.py but for Cosmos-Predict2.5 models.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        s3_credential_path: str = "",
        context_parallel_size: Optional[int] = None,
        model: Optional[Video2WorldInference] = None,
        setup_args: Optional[SetupArguments] = None,
        disable_text_encoder: bool = False,
    ):
        """
        Initialize the feature extractor.

        Args:
            experiment_name (str, optional): Name of the experiment configuration.
                If None, will use default from checkpoint.
            ckpt_path (str, optional): Path to the model checkpoint (local or S3).
                If None, will use default checkpoint.
            s3_credential_path (str): Path to S3 credentials file (if loading from S3).
            context_parallel_size (int, optional): Number of GPUs for context parallelism.
                If None, will use default from setup_args or environment.
            model (Video2WorldInference, optional): Pre-loaded model. If provided,
                will use this instead of loading a new one.
            setup_args (SetupArguments, optional): Setup arguments with model configuration.
                If provided, will use these to determine experiment_name and ckpt_path.
        """
        super().__init__()
        
        if model is not None:
            self.pipe = model
        else:
            # Use setup_args if provided to get default configuration
            if setup_args is not None:
                experiment_name = experiment_name or setup_args.experiment
                ckpt_path = ckpt_path or (
                    str(setup_args.checkpoint_path) if setup_args.checkpoint_path else None
                )
                context_parallel_size = context_parallel_size or setup_args.context_parallel_size or 1

            if experiment_name is None or ckpt_path is None:
                from cosmos_predict2.config import DEFAULT_CHECKPOINT

                experiment_name = experiment_name or DEFAULT_CHECKPOINT.experiment
                ckpt_path = ckpt_path or DEFAULT_CHECKPOINT.s3.uri

            # Load model using the same method as Video2WorldInference
            self.pipe = Video2WorldInference(
                experiment_name=experiment_name,
                ckpt_path=ckpt_path,
                s3_credential_path=s3_credential_path,
                context_parallel_size=context_parallel_size or 1,
                disable_text_encoder=disable_text_encoder,
            )
        
        self.model = self.pipe.model
        self.config = self.pipe.config
        self.batch_size = 1

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        texts: Union[str, List[str]],
        timestep: Union[torch.Tensor, float, int],
        extract_layer_idx: Optional[int] = None,
        all_layers: bool = True,
        step_time: int = 1,
        num_conditional_frames: int = 1,
        negative_prompt: Optional[str] = None,
        use_neg_prompt: bool = False,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract features from the diffusion model.

        Args:
            pixel_values (torch.Tensor): Input video tensor of shape (B, C, T, H, W).
                Should be clean video (not noisy).
            texts (str or List[str]): Text prompt(s) for conditioning.
            timestep (torch.Tensor, float, or int): Timestep for diffusion (0-1 range).
                Lower values mean less noise.
            extract_layer_idx (int, optional): Specific layer index to extract.
                If None and all_layers=True, extracts all layers.
            all_layers (bool): If True, extract features from all DiT blocks and concatenate.
            step_time (int): Number of denoising steps to perform (default: 1).
            num_conditional_frames (int): Number of conditional frames (default: 1).
            negative_prompt (str, optional): Negative prompt for classifier-free guidance.
            use_neg_prompt (bool): Whether to use negative prompt.

        Returns:
            torch.Tensor: Extracted features in 5D format (B, T, H, W, D).
                - If all_layers=True: concatenated features from all blocks along D dimension.
                  Shape: (B, T, H, W, D_total) where D_total = sum of all block feature dimensions.
                - If extract_layer_idx is specified: single block feature.
                  Shape: (B, T, H, W, D) where D is the model's feature dimension.
                - T, H, W are patch-level dimensions (not pixel-level).
        """
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        # Ensure pixel_values is in the correct format
        if pixel_values.dim() == 4:
            # (C, T, H, W) -> (1, C, T, H, W)
            pixel_values = pixel_values.unsqueeze(0)
        B, C, T, H, W = pixel_values.shape

        # Convert text to list if needed
        if texts is None:
            texts = ["a video"] * B
        elif isinstance(texts, str):
            texts = [texts] * B
        else:
            texts = list(texts)
            if len(texts) == 1 and B > 1:
                texts = texts * B
            elif len(texts) != B:
                raise ValueError(
                    f"Number of text prompts ({len(texts)}) must match batch size ({B})."
                )

        precomputed_text_embeddings = None
        if text_embeddings is not None:
            text_embeddings = torch.as_tensor(text_embeddings, device=device)
            if text_embeddings.ndim == 2:
                text_embeddings = text_embeddings.unsqueeze(0)
            if text_embeddings.shape[0] != B:
                raise ValueError(
                    f"Text embeddings batch dimension {text_embeddings.shape[0]} "
                    f"does not match video batch size {B}."
                )
            precomputed_text_embeddings = text_embeddings
        
        # Prepare data batch
        data_batch = self.pipe._get_data_batch_input(
            video=pixel_values,
            prompt=texts,
            num_conditional_frames=num_conditional_frames,
            negative_prompt=negative_prompt or _DEFAULT_NEGATIVE_PROMPT,
            use_neg_prompt=use_neg_prompt,
            precomputed_text_embeddings=precomputed_text_embeddings,
        )

        # Get condition from data batch
        condition, _ = self.model.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)

        # Encode video to latent space using tokenizer
        # The video should be normalized to [-1, 1] range for the tokenizer
        video_normalized = pixel_values.clone()
        if video_normalized.max() > 1.1:  # Assume [0, 255] range
            video_normalized = video_normalized / 127.5 - 1.0
        elif video_normalized.min() >= 0 and video_normalized.max() <= 1.0:  # Assume [0, 1] range
            video_normalized = video_normalized * 2.0 - 1.0
        # If already in [-1, 1] range, use as is

        # Encode to latent
        tensor_kwargs = getattr(
            self.model, "tensor_kwargs", {"device": device, "dtype": video_normalized.dtype}
        )
        tensor_kwargs_fp32 = getattr(
            self.model, "tensor_kwargs_fp32", {**tensor_kwargs, "dtype": torch.float32}
        )
        with torch.no_grad():
            latent_state = self.model.tokenizer.encode(video_normalized.to(**tensor_kwargs))
        # latent_state shape: (B, C_latent, T_latent, H_latent, W_latent)

        # Convert timestep to sigma (noise level)
        # timestep should be in [0, 1] range, where 0 = no noise, 1 = max noise
        if isinstance(timestep, (float, int)):
            t = torch.tensor([timestep], device=device, dtype=torch.float32)
        else:
            t = timestep.to(device=device, dtype=torch.float32)
            if t.ndim == 0:
                t = t.unsqueeze(0)
        
        is_edm_model = hasattr(self.model, "sde")

        if is_edm_model:
            sigma_min = self.model.sde.sigma_min
            sigma_max = self.model.sde.sigma_max
            sigma = sigma_min + t * (sigma_max - sigma_min)

            if sigma.ndim == 1:
                sigma_B = sigma.expand(B)
            else:
                sigma_B = sigma

            if t.item() < 1e-6:
                noisy_latent = latent_state
            else:
                noise = torch.randn_like(latent_state)
                sigma_B_1 = sigma_B.view(B, 1, 1, 1, 1)
                noisy_latent = latent_state + noise * sigma_B_1.to(latent_state.dtype)

            sigma_B_1_T_1_1 = sigma_B.view(B, 1, 1, 1, 1)
            c_skip, c_out, c_in, c_noise = self.model.scaling(sigma=sigma_B_1_T_1_1.to(**tensor_kwargs))
            net_state_in = noisy_latent * c_in
        else:
            sigma_B = sigma_B_1_T_1_1 = None
            c_skip = c_out = c_in = c_noise = None
            noisy_latent = latent_state
            net_state_in = latent_state.to(**tensor_kwargs)

        if condition.is_video and getattr(condition, "gt_frames", None) is not None:
            if is_edm_model:
                condition_state_in = condition.gt_frames.type_as(net_state_in) / self.model.config.sigma_data
            else:
                condition_state_in = condition.gt_frames.type_as(net_state_in)

            if not condition.use_video_condition:
                condition_state_in = condition_state_in * 0

            _, C_latent, _, _, _ = noisy_latent.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, C_latent, 1, 1, 1
            ).type_as(net_state_in)

            net_state_in = condition_state_in * condition_video_mask + net_state_in * (1 - condition_video_mask)

            if is_edm_model:
                sigma_cond = torch.ones_like(sigma_B_1_T_1_1) * self.model.config.sigma_conditional
                _, _, _, c_noise_cond = self.model.scaling(sigma=sigma_cond)
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                c_noise = c_noise_cond * condition_video_mask_B_1_T_1_1 + c_noise * (1 - condition_video_mask_B_1_T_1_1)

        # Get spatial-temporal dimensions before forward pass
        # We need to know T, H, W dimensions to reshape flattened features back to 5D
        # Call prepare_embedded_sequence to get the patch dimensions
        net_state_in_prepared = net_state_in.to(**tensor_kwargs)
        fps = data_batch.get("fps", None)
        padding_mask = data_batch.get("padding_mask", None)
        
        # Move fps and padding_mask to correct device if they exist
        if fps is not None:
            fps = fps.to(**tensor_kwargs)
        if padding_mask is not None:
            padding_mask = padding_mask.to(**tensor_kwargs)
        
        # Get the embedded sequence to determine T, H, W patch dimensions
        prepare_input = ensure_condition_channels(
            net_state_in_prepared, condition, getattr(self.model.net, "in_channels", net_state_in_prepared.shape[1])
        )
        x_B_T_H_W_D_prep, _, _ = self.model.net.prepare_embedded_sequence(
            prepare_input,
            fps=fps,
            padding_mask=padding_mask,
        )
        B_feat, T_feat, H_feat, W_feat, D_feat = x_B_T_H_W_D_prep.shape
        # These dimensions will be used to reshape flattened features back to 5D

        # Determine which layers to extract
        num_blocks = len(self.model.net.blocks)
        if all_layers:
            intermediate_feature_ids = list(range(num_blocks))
        elif extract_layer_idx is not None:
            intermediate_feature_ids = [extract_layer_idx]
        else:
            intermediate_feature_ids = None

        if is_edm_model:
            timestep_kwargs = dict(tensor_kwargs)
            if getattr(self.model.config, "use_wan_fp32_strategy", False):
                timestep_kwargs["dtype"] = torch.float32
            timesteps_B_T = c_noise.squeeze(dim=[1, 3, 4]).to(**timestep_kwargs)
        else:
            t_scalar = float(t.squeeze().clamp(0.0, 1.0).item())
            t_continuous = torch.full((B,), t_scalar, **tensor_kwargs_fp32)
            discrete_ts = self.model.rectified_flow.get_discrete_timestamp(t_continuous, tensor_kwargs_fp32)
            target_dtype = (
                torch.float32 if getattr(self.model.config, "use_wan_fp32_strategy", False) else tensor_kwargs_fp32["dtype"]
            )
            timesteps_B_T = discrete_ts.unsqueeze(1).to(dtype=target_dtype, device=tensor_kwargs["device"])

        condition_kwargs = condition.to_dict()
        mask_key = "condition_video_input_mask_B_C_T_H_W"
        condition_video_mask = getattr(condition, mask_key, None)
        if condition_video_mask is None:
            condition_video_mask = torch.zeros(
                (B, 1, prepare_input.shape[2], prepare_input.shape[3], prepare_input.shape[4]),
                dtype=prepare_input.dtype,
                device=prepare_input.device,
            )
        condition_kwargs[mask_key] = condition_video_mask

        net_output_result = self.model.net(
            x_B_C_T_H_W=net_state_in_prepared,
            timesteps_B_T=timesteps_B_T,
            intermediate_feature_ids=intermediate_feature_ids,
            **condition_kwargs,
        )
        
        # Handle return value: could be tuple (output, features) or just output
        if isinstance(net_output_result, tuple):
            net_output, intermediate_features = net_output_result
        else:
            net_output = net_output_result
            intermediate_features = None

        # Convert flattened features back to 5D format (B, T, H, W, D)
        if intermediate_features and len(intermediate_features) > 0:
            # Reshape each flattened feature (B, L, D) back to (B, T, H, W, D)
            features_5d = []
            for feat in intermediate_features:
                # feat is (B, L, D) where L = T * H * W
                B_f, L_f, D_f = feat.shape
                # Reshape to (B, T, H, W, D)
                feat_5d = rearrange(
                    feat,
                    "b (t h w) d -> b t h w d",
                    t=T_feat,
                    h=H_feat,
                    w=W_feat,
                )
                features_5d.append(feat_5d)
            
            if all_layers and len(features_5d) > 1:
                # Concatenate all features along the feature dimension
                # All features should have the same (B, T, H, W) dimensions
                # Concatenate along D dimension: (B, T, H, W, D1+D2+...+Dn)
                concatenated_features = torch.cat(features_5d, dim=-1)
                return concatenated_features
            else:
                # Return the first (and only) extracted feature in 5D format
                return features_5d[0]
        else:
            # Fallback: return the network output in 5D format
            # net_output is (B, C, T, H, W), we want (B, T, H, W, D)
            # But net_output is the final output after unpatchify, so it's already in spatial format
            # We need to convert it to patch format for consistency
            # Actually, for fallback, we can return it as is or convert to patch format
            # Let's return it in a consistent 5D format by treating C as D
            B_out, C_out, T_out, H_out, W_out = net_output.shape
            # Rearrange to (B, T, H, W, C) format
            net_output_5d = rearrange(net_output, "b c t h w -> b t h w c")
            return net_output_5d

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pipe'):
            self.pipe.cleanup()

