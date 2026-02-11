"""
REINFORCE Policy Network Model.

Policy network that outputs action probabilities for the REINFORCE algorithm.
Unlike DQN which outputs Q-values, this outputs a probability distribution
over actions using softmax.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm.

    Architecture:
    - Input: (N, input_size) flattened representation output
    - Hidden layers: configurable sizes with ReLU activation
    - Output: (N, 4) action logits (converted to probabilities via softmax)

    Per DEC-0037: input_size is dynamic based on representation.
    """

    def __init__(
        self,
        input_size: int,  # Required - from representation.output_shape()[0]
        hidden_layers: List[int] = [256, 256],
        output_size: int = 4,  # 4 actions (up, down, left, right)
        activation: str = "relu"
    ):
        """Initialize policy network.

        Args:
            input_size: Size of flattened input from representation
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

        # Output layer (logits, no activation)
        layers.append(nn.Linear(in_features, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            state: (N, input_size) flattened representation output
                   OR (N, 16, 17) raw one-hot states (legacy support)

        Returns:
            (N, 4) action logits (unnormalized)
        """
        # Flatten input if 3D (legacy raw state input)
        if state.dim() == 3:
            state = state.view(state.size(0), -1)

        # Convert boolean to float if needed
        if state.dtype == torch.bool:
            state = state.float()

        return self.network(state)

    def get_action_probs(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get action probabilities with invalid actions masked.

        Per DEC-0034: Set invalid action probabilities to 0 so they're never selected.

        Args:
            state: (N, 16, 17) or (N, 272) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            (N, 4) action probabilities (sum to 1 over valid actions)
        """
        logits = self.forward(state)

        # Mask invalid actions with very negative value before softmax
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        # Softmax to get probabilities
        probs = F.softmax(masked_logits, dim=1)

        return probs

    def get_action_log_probs(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get log probabilities for gradient computation.

        Args:
            state: (N, 16, 17) or (N, 272) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            (N, 4) action log probabilities
        """
        logits = self.forward(state)

        # Mask invalid actions with very negative value before log_softmax
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        # Log softmax for numerical stability
        log_probs = F.log_softmax(masked_logits, dim=1)

        return log_probs
