"""
Base Representation Interface.

Per DEC-0008: Representation module interface requires:
- __init__(config: dict)
- forward(state: Tensor) -> Tensor
- output_shape() -> tuple

All concrete representations MUST inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import torch.nn as nn
from torch import Tensor


class Representation(nn.Module, ABC):
    """Abstract base class for input representations.

    All representations transform the canonical game state from GameEnv
    (shape: N, 16, 17 one-hot boolean) into a format suitable for the
    RL algorithm's neural network.

    Per DEC-0008: This is the required interface for all representations.

    Attributes:
        config: Configuration dictionary with representation-specific params
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize representation.

        Args:
            config: Configuration dictionary with representation-specific
                    hyperparameters. Contents vary by representation type.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        """Transform game state to representation.

        Args:
            state: (N, 16, 17) one-hot encoded board states from GameEnv
                   dtype is typically boolean

        Returns:
            Transformed representation suitable for algorithm's network.
            Output is always flattened for MLP input (OQ-5 approved).
        """
        pass

    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """Return the output shape of this representation.

        This allows algorithms to determine their input layer size.

        Returns:
            Tuple representing the shape of the output tensor,
            excluding the batch dimension. For example, (272,) for
            flattened one-hot representation.
        """
        pass
