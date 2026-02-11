"""
Expected SARSA Q-Network Model.

Q-network for Expected SARSA - identical to SARSA network.
The difference is in how the target is computed (expected value).

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class ExpectedSARSANetwork(nn.Module):
    """Q-Network for Expected SARSA algorithm.

    Architecture:
    - MLP that processes state representation
    - Outputs Q-values for all actions

    Per DEC-0037: input_size is dynamic based on representation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 256],
        num_actions: int = 4,
        activation: str = "relu"
    ):
        """Initialize Expected SARSA network.

        Args:
            input_size: Size of flattened input from representation
            hidden_layers: List of hidden layer sizes
            num_actions: Number of output actions (4 for 2048)
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        # Build network
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

        # Output layer
        layers.append(nn.Linear(in_features, num_actions))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            state: (N, input_size) or (N, 16, 17) board states

        Returns:
            (N, 4) Q-values for each action
        """
        if state.dim() == 3:
            state = state.view(state.size(0), -1)
        if state.dtype == torch.bool:
            state = state.float()
        return self.network(state)

    def get_action_values(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get Q-values with invalid actions masked.

        Per DEC-0034: Set invalid Q-values to -inf.

        Args:
            state: (N, input_size) or (N, 16, 17) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            (N, 4) Q-values with invalid actions set to -inf
        """
        q_values = self.forward(state)
        masked_q = q_values.clone()
        masked_q[~valid_mask] = float('-inf')
        return masked_q
