"""
Input Representations Module.

This module provides different input representation transformations for the 2048
game state. Each representation takes the canonical (N, 16, 17) one-hot tensor
from GameEnv and transforms it for use by RL algorithms.

Per DEC-0008: All representations implement the interface:
- __init__(config: dict)
- forward(state: Tensor) -> Tensor
- output_shape() -> tuple

Available representations:
- OneHotRepresentation: Pass-through identity (flattens to N, 272)
- EmbeddingRepresentation: Learned embeddings per tile value
- CNNRepresentation: Configurable convolutional encoder
"""

from representations.base import Representation
from representations.onehot import OneHotRepresentation
from representations.embedding import EmbeddingRepresentation
from representations.cnn import CNNRepresentation

__all__ = [
    "Representation",
    "OneHotRepresentation",
    "EmbeddingRepresentation",
    "CNNRepresentation",
]
