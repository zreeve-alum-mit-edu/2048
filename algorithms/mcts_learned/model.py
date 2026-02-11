"""
MCTS+Learned Network Model.

Implements a policy-value network similar to AlphaZero/PPO that outputs
both a policy (action probabilities) and value estimate for guiding MCTS.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PolicyValueNetwork(nn.Module):
    """Policy-Value Network for MCTS+learned.

    Architecture similar to PPO's ActorCriticNetwork:
    - Shared trunk: MLP that processes state representation
    - Policy head: outputs action logits
    - Value head: outputs scalar value estimate

    Used by MCTS for:
    - Prior probabilities P(s,a) for UCB exploration
    - Value estimates V(s) for leaf evaluation
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 256],
        num_actions: int = 4,
        activation: str = "relu"
    ):
        """Initialize policy-value network.

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

        # Policy head - outputs action logits
        self.policy_head = nn.Sequential(
            nn.Linear(self.trunk_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        # Value head - outputs scalar value
        self.value_head = nn.Sequential(
            nn.Linear(self.trunk_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through the network.

        Args:
            state: (N, input_size) flattened representation output
                   OR (N, 16, 17) raw one-hot states (legacy support)

        Returns:
            Tuple of:
            - policy_logits: (N, 4) unnormalized action scores
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

        # Policy and value heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        return policy_logits, value

    def get_policy(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get action probabilities with invalid actions masked.

        Per DEC-0034: Set invalid action probabilities to 0.

        Args:
            state: (N, 16, 17) or (N, input_size) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            (N, 4) action probabilities (sum to 1 over valid actions)
        """
        logits, _ = self.forward(state)

        # Mask invalid actions with very negative value before softmax
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        # Softmax to get probabilities
        probs = F.softmax(masked_logits, dim=1)

        return probs

    def get_policy_and_value(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Get both policy and value, with policy masked.

        Args:
            state: (N, 16, 17) or (N, input_size) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            Tuple of:
            - policy: (N, 4) action probabilities
            - value: (N,) state value estimates
        """
        logits, value = self.forward(state)

        # Mask invalid actions
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        # Softmax to get probabilities
        policy = F.softmax(masked_logits, dim=1)

        return policy, value

    def get_value(self, state: Tensor) -> Tensor:
        """Get only value estimate.

        Args:
            state: (N, 16, 17) or (N, input_size) board states

        Returns:
            (N,) state value estimates
        """
        _, value = self.forward(state)
        return value

    def get_log_policy_and_value(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Get log policy and value for loss computation.

        Args:
            state: (N, 16, 17) or (N, input_size) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            Tuple of:
            - log_policy: (N, 4) log action probabilities
            - value: (N,) state value estimates
        """
        logits, value = self.forward(state)

        # Mask invalid actions
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        # Log softmax for numerical stability
        log_policy = F.log_softmax(masked_logits, dim=1)

        return log_policy, value
