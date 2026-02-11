"""
QR-DQN Network Model.

Implements Quantile Regression DQN that learns N fixed quantiles of the
return distribution using quantile regression loss.

Unlike C51 which uses fixed atom locations, QR-DQN uses fixed quantile
fractions and learns the quantile values.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class QRNetwork(nn.Module):
    """QR-DQN (Quantile Regression DQN) Network.

    Architecture:
    - Shared trunk: MLP that processes state representation
    - Output layer: (num_actions * n_quantiles) values
    - Reshaped to (batch, num_actions, n_quantiles)

    The network outputs quantile values for each action.
    Action values are computed as the mean of quantiles.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 256],
        num_actions: int = 4,
        n_quantiles: int = 200,
        activation: str = "relu"
    ):
        """Initialize QR-DQN network.

        Args:
            input_size: Size of flattened input from representation
            hidden_layers: List of hidden layer sizes for trunk
            num_actions: Number of output actions
            n_quantiles: Number of quantiles to estimate
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions
        self.n_quantiles = n_quantiles

        # Compute quantile midpoints: tau_i = (i + 0.5) / N
        self.register_buffer(
            'taus',
            (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        )

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

        # Output layer: produces quantile values for all (action, quantile) pairs
        self.output_layer = nn.Linear(
            self.trunk_output_size,
            num_actions * n_quantiles
        )

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            state: (N, input_size) flattened representation output
                   OR (N, 16, 17) raw one-hot states (legacy support)

        Returns:
            (N, num_actions, n_quantiles) quantile values
        """
        # Flatten input if 3D
        if state.dim() == 3:
            state = state.view(state.size(0), -1)

        # Convert boolean to float if needed
        if state.dtype == torch.bool:
            state = state.float()

        # Forward through trunk
        features = self.trunk(state)

        # Get quantile values and reshape to (batch, actions, quantiles)
        quantile_values = self.output_layer(features)
        quantile_values = quantile_values.view(-1, self.num_actions, self.n_quantiles)

        return quantile_values

    def get_q_values(self, state: Tensor) -> Tensor:
        """Compute Q-values as mean of quantiles.

        Q(s,a) = mean(quantile_values_a)

        Args:
            state: (N, 16, 17) or (N, input_size) board states

        Returns:
            (N, 4) Q-values for each action
        """
        quantile_values = self.forward(state)  # (N, 4, n_quantiles)
        # Mean over quantiles
        q_values = quantile_values.mean(dim=2)
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

    def get_quantiles(self, state: Tensor) -> Tensor:
        """Get full quantile values for all actions.

        Args:
            state: (N, 16, 17) or (N, input_size) board states

        Returns:
            (N, 4, n_quantiles) quantile values
        """
        return self.forward(state)
