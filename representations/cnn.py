"""
CNN Representation.

This representation uses configurable convolutional layers to extract
spatial features from the 2048 board. Supports:
- Square and rectangular kernels (4x1, 1x4, 2x2, 3x3, etc.)
- Inception-style multi-kernel configurations (multiple kernel sizes in parallel)
- Configurable strides and output channels

Per OQ-1: No batch normalization.
Per OQ-3: No residual connections.
Per OQ-4: Default CNN config is single 2x2 kernel.
Per OQ-5: Output is always flat for MLP input.

Per design docs section 8.4, kernel shapes capture different patterns:
- 4x1: Full rows
- 1x4: Full columns
- 4x2: Half-board horizontal slices
- 2x4: Half-board vertical slices
- 3x3: Corner/center regions
- 2x2: Quadrants
"""

from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from representations.base import Representation


class CNNRepresentation(Representation):
    """CNN representation with configurable kernels.

    Input: (N, 16, 17) one-hot boolean from GameEnv
    Output: (N, total_features) flattened float tensor

    The input is reshaped to (N, 17, 4, 4) treating the 17 possible tile
    values as channels and the 16 positions as a 4x4 spatial grid.

    Supports Inception-style multi-kernel configurations where multiple
    kernel sizes are applied in parallel and their outputs concatenated.
    """

    # Default configuration: single 2x2 kernel (OQ-4)
    DEFAULT_CONFIG = {
        "kernels": [
            {"size": [2, 2], "out_channels": 64, "stride": [1, 1]}
        ],
        "combine": "concat",  # How to combine multi-kernel outputs
        "activation": "relu",
    }

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize CNN representation.

        Args:
            config: Configuration dictionary with:
                - kernels (list): List of kernel configs, each with:
                    - size: [height, width] of kernel
                    - out_channels: Number of output channels
                    - stride: [height, width] stride (default [1, 1])
                - combine (str): How to combine outputs ("concat" or "sum")
                - activation (str): Activation function ("relu" or "tanh")

        Default config (OQ-4): Single 2x2 kernel with 64 channels.
        """
        # Merge with defaults
        merged_config = self.DEFAULT_CONFIG.copy()
        if config:
            merged_config.update(config)
        super().__init__(merged_config)

        # Input: 17 channels (one-hot tile values)
        self.in_channels = 17

        # Parse activation
        activation_name = self.config.get("activation", "relu")
        if activation_name == "relu":
            self.activation = nn.ReLU()
        elif activation_name == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

        # Build convolutional branches (Inception-style)
        self.branches = nn.ModuleList()
        self.branch_output_sizes: List[int] = []

        kernel_configs = self.config.get("kernels", self.DEFAULT_CONFIG["kernels"])

        for kernel_config in kernel_configs:
            kernel_size = tuple(kernel_config["size"])
            out_channels = kernel_config["out_channels"]
            stride = tuple(kernel_config.get("stride", [1, 1]))

            # Create conv layer
            conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0  # No padding - preserve spatial meaning
            )
            self.branches.append(conv)

            # Compute output spatial size for this branch
            # Input spatial: 4x4
            out_h = (4 - kernel_size[0]) // stride[0] + 1
            out_w = (4 - kernel_size[1]) // stride[1] + 1
            branch_output_size = out_channels * out_h * out_w
            self.branch_output_sizes.append(branch_output_size)

        # Compute total output size
        self.combine_method = self.config.get("combine", "concat")
        if self.combine_method == "concat":
            self._output_dim = sum(self.branch_output_sizes)
        elif self.combine_method == "sum":
            # All branches must have same output size for sum
            if len(set(self.branch_output_sizes)) > 1:
                raise ValueError(
                    f"Cannot use 'sum' combine with different branch output sizes: "
                    f"{self.branch_output_sizes}"
                )
            self._output_dim = self.branch_output_sizes[0] if self.branch_output_sizes else 0
        else:
            raise ValueError(f"Unknown combine method: {self.combine_method}")

    def forward(self, state: Tensor) -> Tensor:
        """Transform one-hot state through CNN.

        Args:
            state: (N, 16, 17) one-hot encoded board states

        Returns:
            (N, total_features) flattened CNN features
        """
        batch_size = state.size(0)

        # Convert to float if boolean
        if state.dtype == torch.bool:
            state = state.float()

        # Reshape: (N, 16, 17) -> (N, 17, 4, 4)
        # Treat 17 tile values as channels, 16 positions as 4x4 spatial grid
        # First transpose to (N, 17, 16), then reshape to (N, 17, 4, 4)
        x = state.transpose(1, 2).contiguous()  # (N, 17, 16)
        x = x.view(batch_size, 17, 4, 4)  # (N, 17, 4, 4)

        # Apply each branch
        branch_outputs = []
        for conv in self.branches:
            out = conv(x)  # (N, out_channels, out_h, out_w)
            out = self.activation(out)
            out = out.view(batch_size, -1)  # Flatten spatial dims
            branch_outputs.append(out)

        # Combine branch outputs
        if len(branch_outputs) == 1:
            combined = branch_outputs[0]
        elif self.combine_method == "concat":
            combined = torch.cat(branch_outputs, dim=1)
        elif self.combine_method == "sum":
            combined = sum(branch_outputs)
        else:
            combined = branch_outputs[0]  # Fallback

        return combined

    def output_shape(self) -> Tuple[int, ...]:
        """Return output shape.

        Returns:
            (total_features,) - total flattened CNN features
        """
        return (self._output_dim,)
