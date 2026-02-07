"""
Embedding Representation.

This representation uses a learned embedding layer to transform tile values
into dense vectors. A single embedding layer is shared across all 16 positions
(per OQ-2: Yes, shared embedding weights across positions).

Per OQ-1: No batch normalization.
Per OQ-3: No residual connections.
Per OQ-5: Output is always flat for MLP input.
"""

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from representations.base import Representation


class EmbeddingRepresentation(Representation):
    """Embedding representation with shared weights across positions.

    Input: (N, 16, 17) one-hot boolean from GameEnv
    Output: (N, 16 * embed_dim) flattened float tensor

    Uses nn.Embedding(17, embed_dim) to learn dense representations for
    each of the 17 possible tile values (0=empty, 1-16 = 2^1 to 2^16).

    The same embedding is shared across all 16 board positions (OQ-2).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize embedding representation.

        Args:
            config: Configuration dictionary with:
                - embed_dim (int): Embedding dimension per tile value.
                                   Required parameter.
        """
        super().__init__(config)

        # Extract config
        self.embed_dim = config.get("embed_dim", 32)  # Default 32 if not specified

        # Create shared embedding layer
        # 17 possible values (0=empty, 1-16 = tile values)
        self.embedding = nn.Embedding(
            num_embeddings=17,
            embedding_dim=self.embed_dim
        )

        # Pre-compute output dimension
        self._output_dim = 16 * self.embed_dim

    def forward(self, state: Tensor) -> Tensor:
        """Transform one-hot state to embeddings.

        Args:
            state: (N, 16, 17) one-hot encoded board states

        Returns:
            (N, 16 * embed_dim) flattened embedded representation
        """
        batch_size = state.size(0)

        # Convert one-hot to indices: (N, 16, 17) -> (N, 16)
        # argmax along the last dimension gives the tile value index
        if state.dtype == torch.bool:
            state = state.float()
        indices = state.argmax(dim=-1)  # (N, 16)

        # Apply embedding: (N, 16) -> (N, 16, embed_dim)
        embedded = self.embedding(indices)

        # Flatten: (N, 16, embed_dim) -> (N, 16 * embed_dim) per OQ-5
        flat = embedded.view(batch_size, -1)

        return flat

    def output_shape(self) -> Tuple[int, ...]:
        """Return output shape.

        Returns:
            (16 * embed_dim,) - flattened embedding dimension
        """
        return (self._output_dim,)
