"""
C51 Network Model.

Implements the Categorical DQN architecture that outputs a distribution
over returns for each action, discretized into N atoms.

Instead of Q(s,a) as a scalar, outputs Z(s,a) as a categorical distribution
over a fixed support [V_min, V_max].

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class C51Network(nn.Module):
    """C51 (Categorical DQN) Network.

    Architecture:
    - Shared trunk: MLP that processes state representation
    - Output layer: (num_actions * n_atoms) logits
    - Reshaped to (batch, num_actions, n_atoms) probabilities via softmax

    The network outputs a probability distribution over atoms for each action.
    Action values are computed as the expected value: sum(p_i * z_i)
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 256],
        num_actions: int = 4,
        n_atoms: int = 51,
        v_min: float = 0.0,
        v_max: float = 100000.0,
        activation: str = "relu"
    ):
        """Initialize C51 network.

        Args:
            input_size: Size of flattened input from representation
            hidden_layers: List of hidden layer sizes for trunk
            num_actions: Number of output actions
            n_atoms: Number of atoms in the distribution support
            v_min: Minimum value of the support
            v_max: Maximum value of the support
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Compute support: evenly spaced atoms from v_min to v_max
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Build trunk
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

        # Output layer: produces logits for all (action, atom) pairs
        self.output_layer = nn.Linear(
            self.trunk_output_size,
            num_actions * n_atoms
        )

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            state: (N, input_size) flattened representation output
                   OR (N, 16, 17) raw one-hot states (legacy support)

        Returns:
            (N, num_actions, n_atoms) probability distributions
        """
        # Flatten input if 3D
        if state.dim() == 3:
            state = state.view(state.size(0), -1)

        # Convert boolean to float if needed
        if state.dtype == torch.bool:
            state = state.float()

        # Forward through trunk
        features = self.trunk(state)

        # Get logits and reshape to (batch, actions, atoms)
        logits = self.output_layer(features)
        logits = logits.view(-1, self.num_actions, self.n_atoms)

        # Apply softmax over atoms dimension to get probabilities
        probs = F.softmax(logits, dim=2)

        return probs

    def get_q_values(self, state: Tensor) -> Tensor:
        """Compute Q-values as expected value of distributions.

        Q(s,a) = sum_i(p_i * z_i)

        Args:
            state: (N, 16, 17) or (N, input_size) board states

        Returns:
            (N, 4) Q-values for each action
        """
        probs = self.forward(state)  # (N, 4, n_atoms)
        # Compute expected value: sum over atoms
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
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
        q_values = self.get_q_values(state)

        # Mask invalid actions with -inf
        masked_q_values = q_values.clone()
        masked_q_values[~valid_mask] = float('-inf')

        return masked_q_values

    def get_distribution(self, state: Tensor) -> Tensor:
        """Get full probability distributions for all actions.

        Args:
            state: (N, 16, 17) or (N, input_size) board states

        Returns:
            (N, 4, n_atoms) probability distributions
        """
        return self.forward(state)
