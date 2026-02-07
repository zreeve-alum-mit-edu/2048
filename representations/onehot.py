"""
One-Hot Representation (Pass-Through).

This representation simply flattens the canonical (N, 16, 17) one-hot
game state to (N, 272) for direct use by MLP networks.

Per OQ-5: Output is always flat for MLP input.
"""

from typing import Dict, Any, Tuple

import torch
from torch import Tensor

from representations.base import Representation


class OneHotRepresentation(Representation):
    """Pass-through representation that flattens one-hot state.

    Input: (N, 16, 17) one-hot boolean from GameEnv
    Output: (N, 272) flattened float tensor

    This is the simplest representation - just flatten and convert to float.
    No learnable parameters.
    """

    # Output dimension: 16 positions * 17 possible values
    OUTPUT_DIM = 16 * 17  # 272

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize one-hot representation.

        Args:
            config: Optional configuration (not used, kept for interface compliance)
        """
        super().__init__(config or {})

    def forward(self, state: Tensor) -> Tensor:
        """Flatten one-hot state.

        Args:
            state: (N, 16, 17) one-hot encoded board states

        Returns:
            (N, 272) flattened float tensor
        """
        batch_size = state.size(0)

        # Flatten: (N, 16, 17) -> (N, 272)
        flat = state.view(batch_size, -1)

        # Convert to float if boolean
        if flat.dtype == torch.bool:
            flat = flat.float()

        return flat

    def output_shape(self) -> Tuple[int, ...]:
        """Return output shape.

        Returns:
            (272,) - flattened one-hot dimension
        """
        return (self.OUTPUT_DIM,)
