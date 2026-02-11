"""
PPO Actor-Critic Network Model.

Reuses the ActorCriticNetwork architecture from A2C.
PPO's key differences are in the loss computation (agent), not the network architecture.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PPOActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO algorithm.

    Architecture (same as A2C):
    - Shared trunk: MLP that processes state representation
    - Actor head: outputs action logits (policy)
    - Critic head: outputs scalar value estimate

    Per DEC-0037: input_size is dynamic based on representation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 256],
        num_actions: int = 4,
        activation: str = "relu"
    ):
        """Initialize PPO actor-critic network.

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

        # Actor head (policy) - outputs action logits
        self.actor_head = nn.Linear(self.trunk_output_size, num_actions)

        # Critic head (value) - outputs scalar value
        self.critic_head = nn.Linear(self.trunk_output_size, 1)

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
        # Flatten input if 3D (legacy raw state input)
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

    def get_action_log_probs_and_value(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Get log probabilities and value for actor-critic update.

        Args:
            state: (N, 16, 17) or (N, input_size) board states
            valid_mask: (N, 4) boolean mask of valid actions

        Returns:
            Tuple of:
            - log_probs: (N, 4) action log probabilities
            - value: (N,) state value estimates
        """
        logits, value = self.forward(state)

        # Mask invalid actions before log_softmax
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        # Log softmax for numerical stability
        log_probs = F.log_softmax(masked_logits, dim=1)

        return log_probs, value

    def get_value(self, state: Tensor) -> Tensor:
        """Get only value estimate (for bootstrapping).

        Args:
            state: (N, 16, 17) or (N, input_size) board states

        Returns:
            (N,) state value estimates
        """
        _, value = self.forward(state)
        return value

    def evaluate_actions(
        self,
        state: Tensor,
        actions: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions for PPO loss computation.

        Returns log probs of specific actions, values, and entropy.

        Args:
            state: (N, input_size) states
            actions: (N,) actions to evaluate
            valid_mask: (N, 4) valid action masks

        Returns:
            Tuple of:
            - action_log_probs: (N,) log prob of taken actions
            - values: (N,) state values
            - entropy: (N,) policy entropy per state
        """
        logits, values = self.forward(state)

        # Mask invalid actions
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        # Get log probs and probs
        log_probs = F.log_softmax(masked_logits, dim=1)
        probs = F.softmax(masked_logits, dim=1)

        # Log prob of taken action
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Entropy: -sum(p * log(p)) over valid actions only
        # Handle numerical issues with masked actions
        entropy = -(probs * log_probs).sum(dim=1)
        # Clamp entropy to avoid NaN from -inf * 0
        entropy = torch.where(entropy.isfinite(), entropy, torch.zeros_like(entropy))

        return action_log_probs, values, entropy
