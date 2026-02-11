"""
Dueling DQN Network Model.

Implements the dueling architecture which separates:
- Value stream: V(s) - how good is the state
- Advantage stream: A(s,a) - how good is each action relative to others

Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

This allows the network to learn which states are valuable without having
to learn the effect of each action at that state.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN network with separate value and advantage streams.

    Architecture:
    - Shared trunk: MLP that processes state representation
    - Value stream: outputs single scalar V(s)
    - Advantage stream: outputs A(s,a) for each action
    - Combined: Q(s,a) = V(s) + (A(s,a) - mean(A))

    Per DEC-0037: input_size is dynamic based on representation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 256],
        num_actions: int = 4,
        activation: str = "relu"
    ):
        """Initialize Dueling DQN network.

        Args:
            input_size: Size of flattened input from representation
            hidden_layers: List of hidden layer sizes for shared trunk
            num_actions: Number of output actions
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        # Build shared trunk
        trunk_layers = []
        in_features = input_size

        for hidden_size in hidden_layers:
            trunk_layers.append(nn.Linear(in_features, hidden_size))
            if activation == "relu":
                trunk_layers.append(nn.ReLU())
            elif activation == "tanh":
                trunk_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_features = hidden_size

        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_output_size = in_features

        # Value stream: outputs single value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.trunk_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream: outputs A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.trunk_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the network.

        Computes: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

        The mean subtraction ensures identifiability - without it, we could
        add any constant to V and subtract from A without changing Q.

        Args:
            state: (N, input_size) flattened representation output
                   OR (N, 16, 17) raw one-hot states (legacy support)

        Returns:
            (N, 4) Q-values for each action
        """
        # Flatten input if 3D (legacy raw state input)
        if state.dim() == 3:
            state = state.view(state.size(0), -1)

        # Convert boolean to float if needed
        if state.dtype == torch.bool:
            state = state.float()

        # Shared trunk
        features = self.trunk(state)

        # Value and advantage streams
        value = self.value_stream(features)  # (N, 1)
        advantages = self.advantage_stream(features)  # (N, 4)

        # Combine: Q = V + (A - mean(A))
        # Subtracting mean ensures V represents the value and A represents
        # only the relative advantage of each action
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

    def get_action_values(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get Q-values with invalid actions masked.

        Per DEC-0034: Set invalid Q-values to -inf so they're never selected.

        Args:
            state: (N, 16, 17) or (N, input_size) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            (N, 4) Q-values with invalid actions set to -inf
        """
        q_values = self.forward(state)

        # Mask invalid actions with -inf
        masked_q_values = q_values.clone()
        masked_q_values[~valid_mask] = float('-inf')

        return masked_q_values

    def get_value(self, state: Tensor) -> Tensor:
        """Get only the value V(s) without computing full Q-values.

        Useful for analysis/debugging.

        Args:
            state: (N, 16, 17) or (N, input_size) board states

        Returns:
            (N,) state values
        """
        if state.dim() == 3:
            state = state.view(state.size(0), -1)
        if state.dtype == torch.bool:
            state = state.float()

        features = self.trunk(state)
        value = self.value_stream(features).squeeze(-1)
        return value

    def get_advantage(self, state: Tensor) -> Tensor:
        """Get only the advantages A(s,a) without value.

        Useful for analysis/debugging.

        Args:
            state: (N, 16, 17) or (N, input_size) board states

        Returns:
            (N, 4) advantage values (before mean subtraction)
        """
        if state.dim() == 3:
            state = state.view(state.size(0), -1)
        if state.dtype == torch.bool:
            state = state.float()

        features = self.trunk(state)
        advantages = self.advantage_stream(features)
        return advantages
