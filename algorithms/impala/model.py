"""
IMPALA Actor-Critic Network Model.

Network for IMPALA algorithm with:
- Shared feature extraction trunk
- Actor head (policy logits)
- Critic head (state value V(s))

IMPALA uses V(s) like A2C/A3C, but with V-trace correction for
off-policy learning.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IMPALANetwork(nn.Module):
    """Actor-Critic network for IMPALA algorithm.

    Architecture:
    - Shared trunk: MLP that processes state representation
    - Actor head: outputs action logits (policy)
    - Critic head: outputs scalar value V(s)

    Per DEC-0037: input_size is dynamic based on representation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 256],
        num_actions: int = 4,
        activation: str = "relu"
    ):
        """Initialize IMPALA network.

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

        # Critic head (value) - outputs scalar value
        self.critic_head = nn.Linear(self.trunk_output_size, 1)

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
            - action_logits: (N, 4) unnormalized action scores
            - value: (N,) state value estimates
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
        value = self.critic_head(features).squeeze(-1)

        return action_logits, value

    def get_action_probs(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get action probabilities with invalid actions masked.

        Per DEC-0034: Set invalid action probabilities to 0.

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

    def get_policy_logits_and_value(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Get policy logits and value.

        Args:
            state: (N, input_size) or (N, 16, 17) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            Tuple of:
            - logits: (N, 4) action logits (masked)
            - value: (N,) state value estimates
        """
        logits, value = self.forward(state)

        # Mask invalid actions
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        return masked_logits, value

    def get_action_log_probs_and_value(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Get log probabilities and value.

        Args:
            state: (N, input_size) or (N, 16, 17) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            Tuple of:
            - log_probs: (N, 4) action log probabilities
            - value: (N,) state value estimates
        """
        logits, value = self.get_policy_logits_and_value(state, valid_mask)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs, value

    def get_value(self, state: Tensor) -> Tensor:
        """Get only value estimate.

        Args:
            state: (N, input_size) or (N, 16, 17) board states

        Returns:
            (N,) state value estimates
        """
        _, value = self.forward(state)
        return value
