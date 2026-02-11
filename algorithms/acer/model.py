"""
ACER Actor-Critic Network Model.

Network for ACER algorithm with:
- Shared feature extraction trunk
- Actor head (policy) with action probabilities
- Critic head (Q-values for all actions)

ACER requires Q-values for all actions to compute importance-weighted
corrections, unlike A2C which only needs V(s).

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ACERNetwork(nn.Module):
    """Actor-Critic network for ACER algorithm.

    Architecture:
    - Shared trunk: MLP that processes state representation
    - Actor head: outputs action logits (policy pi(a|s))
    - Critic head: outputs Q-values for all actions Q(s,a)

    Unlike A2C/A3C which use V(s), ACER needs Q(s,a) for all actions
    to compute the Retrace correction.

    Per DEC-0037: input_size is dynamic based on representation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 256],
        num_actions: int = 4,
        activation: str = "relu"
    ):
        """Initialize ACER network.

        Args:
            input_size: Size of flattened input from representation
            hidden_layers: List of hidden layer sizes for shared trunk
            num_actions: Number of output actions (4 for 2048)
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

        # Actor head (policy) - outputs action logits
        self.actor_head = nn.Linear(self.trunk_output_size, num_actions)

        # Critic head (Q-values) - outputs Q for each action
        self.critic_head = nn.Linear(self.trunk_output_size, num_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through the network.

        Args:
            state: (N, input_size) flattened representation output
                   OR (N, 16, 17) raw one-hot states (legacy support)

        Returns:
            Tuple of:
            - action_logits: (N, 4) unnormalized action scores (policy)
            - q_values: (N, 4) Q-values for each action
        """
        # Flatten input if 3D
        if state.dim() == 3:
            state = state.view(state.size(0), -1)

        # Convert boolean to float if needed
        if state.dtype == torch.bool:
            state = state.float()

        # Shared trunk
        features = self.trunk(state)

        # Actor and critic heads
        action_logits = self.actor_head(features)
        q_values = self.critic_head(features)

        return action_logits, q_values

    def get_policy_and_q(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get policy probabilities, log probs, and Q-values.

        Per DEC-0034: Invalid actions are masked.

        Args:
            state: (N, input_size) or (N, 16, 17) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            Tuple of:
            - probs: (N, 4) action probabilities
            - log_probs: (N, 4) action log probabilities
            - q_values: (N, 4) Q-values
        """
        logits, q_values = self.forward(state)

        # Mask invalid actions
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        # Get probabilities and log probabilities
        log_probs = F.log_softmax(masked_logits, dim=1)
        probs = F.softmax(masked_logits, dim=1)

        return probs, log_probs, q_values

    def get_action_probs(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get action probabilities with invalid actions masked.

        Args:
            state: (N, input_size) or (N, 16, 17) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            (N, 4) action probabilities
        """
        logits, _ = self.forward(state)

        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        probs = F.softmax(masked_logits, dim=1)
        return probs

    def get_value(self, state: Tensor, valid_mask: Tensor) -> Tensor:
        """Get state value V(s) = sum_a pi(a|s) * Q(s,a).

        Args:
            state: (N, input_size) or (N, 16, 17) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            (N,) state value estimates
        """
        probs, _, q_values = self.get_policy_and_q(state, valid_mask)
        # V(s) = E_a[Q(s,a)] under policy pi
        value = (probs * q_values).sum(dim=1)
        return value
