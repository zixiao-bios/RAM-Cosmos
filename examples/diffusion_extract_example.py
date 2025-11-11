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
Example usage of Diffusion_feature_extractor for extracting features from Cosmos-Predict2.5 model.
"""

import torch
from pathlib import Path
from cosmos_predict2.config import SetupArguments
from cosmos_predict2._src.predict2.inference.diffusion_extract import Diffusion_feature_extractor
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io


def main():
    """Example of extracting features from a video."""
    
    # Initialize with default configuration (same as inference.py)
    # This will use the default 2B post-trained model
    setup_args = SetupArguments(
        output_dir=Path("outputs/feature_extraction"),
        model="2B/post-trained",  # Use default model
    )
    
    # Create feature extractor using setup_args for default configuration
    extractor = Diffusion_feature_extractor(
        setup_args=setup_args,  # This will automatically use the correct experiment and checkpoint
    )
    
    # Load a video (example: load from file)
    # video_path = "path/to/your/video.mp4"
    # video_frames, _ = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
    # video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1]
    # video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    # video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: (1, C, T, H, W)
    
    # For demonstration, create a dummy video tensor
    # Shape: (B, C, T, H, W) = (1, 3, 16, 256, 256)
    video_tensor = torch.randn(1, 3, 16, 256, 256)
    
    # Text prompt
    text_prompt = "A point-of-view video shot from inside a vehicle."
    
    # Extract features
    # timestep: 0.5 means medium noise level (0 = no noise, 1 = max noise)
    features = extractor(
        pixel_values=video_tensor,
        texts=text_prompt,
        timestep=0.5,
        all_layers=True,  # Extract from all DiT blocks
        num_conditional_frames=1,
    )
    
    print(f"Extracted features shape: {features.shape}")
    # Expected shape: (B, T, H, W, D_total) where:
    # - B: batch size
    # - T: temporal patch dimension (number of temporal patches)
    # - H: height patch dimension (number of height patches)
    # - W: width patch dimension (number of width patches)
    # - D_total: total feature dimension (sum of all block feature dimensions if all_layers=True)
    # Note: T, H, W are patch-level dimensions, not pixel-level dimensions
    
    # Cleanup
    extractor.cleanup()
    
    return features


if __name__ == "__main__":
    main()

