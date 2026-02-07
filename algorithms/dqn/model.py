"""
DQN Network Model.

MLP network that takes flattened one-hot state (N, 16, 17) -> (N, 272)
and outputs Q-values for each action (N, 4).
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class DQNNetwork(nn.Module):
    """DQN network with MLP architecture.

    Architecture:
    - Input: (N, 272) flattened one-hot state
    - Hidden layers: configurable sizes with ReLU activation
    - Output: (N, 4) Q-values for each action
    """

    def __init__(
        self,
        input_size: int = 272,  # 16 positions * 17 values
        hidden_layers: List[int] = [256, 256],
        output_size: int = 4,  # 4 actions (up, down, left, right)
        activation: str = "relu"
    ):
        """Initialize DQN network.

        Args:
            input_size: Size of flattened input (default 272 for 16*17)
            hidden_layers: List of hidden layer sizes
            output_size: Number of output actions
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Build network layers
        layers = []
        in_features = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_features = hidden_size

        # Output layer (no activation)
        layers.append(nn.Linear(in_features, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            state: (N, 16, 17) one-hot encoded board states
                   OR (N, 272) flattened states

        Returns:
            (N, 4) Q-values for each action
        """
        # Flatten input if needed
        if state.dim() == 3:
            # (N, 16, 17) -> (N, 272)
            state = state.view(state.size(0), -1)

        # Convert boolean to float if needed
        if state.dtype == torch.bool:
            state = state.float()

        return self.network(state)

    def get_action_values(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get Q-values with invalid actions masked.

        Per DEC-0034: Set invalid Q-values to -inf so they're never selected.

        Args:
            state: (N, 16, 17) or (N, 272) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            (N, 4) Q-values with invalid actions set to -inf
        """
        q_values = self.forward(state)

        # Mask invalid actions with -inf
        masked_q_values = q_values.clone()
        masked_q_values[~valid_mask] = float('-inf')

        return masked_q_values
