"""
MuZero-style Network Models.

Implements the three networks used by MuZero:
1. Representation Network: h(observation) -> hidden_state
2. Dynamics Network: g(hidden_state, action) -> (next_hidden_state, reward)
3. Prediction Network: f(hidden_state) -> (policy, value)

This is a simplified version for the 2048 game.
Per DEC-0037: Input size is dynamic based on representation output shape.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RepresentationNetwork(nn.Module):
    """Maps observations to hidden states.

    h(observation) -> hidden_state
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: List[int] = [256],
    ):
        """Initialize representation network.

        Args:
            input_size: Size of flattened input (e.g., 16*17=272)
            hidden_size: Size of output hidden state
            hidden_layers: Hidden layer sizes
        """
        super().__init__()

        layers = []
        in_features = input_size

        for layer_size in hidden_layers:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.ReLU())
            in_features = layer_size

        layers.append(nn.Linear(in_features, hidden_size))

        self.network = nn.Sequential(*layers)
        self.hidden_size = hidden_size

    def forward(self, observation: Tensor) -> Tensor:
        """Encode observation to hidden state.

        Args:
            observation: (N, input_size) or (N, 16, 17) game state

        Returns:
            (N, hidden_size) hidden state
        """
        if observation.dim() == 3:
            observation = observation.view(observation.size(0), -1)
        if observation.dtype == torch.bool:
            observation = observation.float()

        hidden_state = self.network(observation)

        # Normalize hidden state (as in MuZero paper)
        hidden_state = self._normalize(hidden_state)

        return hidden_state

    def _normalize(self, x: Tensor) -> Tensor:
        """Min-max normalize to [0, 1]."""
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        range_val = x_max - x_min
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        return (x - x_min) / range_val


class DynamicsNetwork(nn.Module):
    """Predicts next hidden state and reward.

    g(hidden_state, action) -> (next_hidden_state, reward)
    """

    def __init__(
        self,
        hidden_size: int,
        num_actions: int = 4,
        hidden_layers: List[int] = [256],
    ):
        """Initialize dynamics network.

        Args:
            hidden_size: Size of hidden state
            num_actions: Number of possible actions
            hidden_layers: Hidden layer sizes
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # Input: hidden_state concatenated with one-hot action
        input_size = hidden_size + num_actions

        layers = []
        in_features = input_size

        for layer_size in hidden_layers:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.ReLU())
            in_features = layer_size

        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.next_state_head = nn.Linear(in_features, hidden_size)
        self.reward_head = nn.Linear(in_features, 1)

    def forward(
        self,
        hidden_state: Tensor,
        action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Predict next state and reward.

        Args:
            hidden_state: (N, hidden_size) current hidden state
            action: (N,) or (N, 1) action indices

        Returns:
            Tuple of:
            - next_hidden_state: (N, hidden_size)
            - reward: (N,) predicted reward
        """
        batch_size = hidden_state.size(0)

        # Create one-hot action encoding
        if action.dim() == 1:
            action = action.unsqueeze(1)
        action_onehot = torch.zeros(
            batch_size, self.num_actions,
            device=hidden_state.device
        )
        action_onehot.scatter_(1, action, 1)

        # Concatenate state and action
        x = torch.cat([hidden_state, action_onehot], dim=1)

        # Forward through trunk
        features = self.trunk(x)

        # Get outputs
        next_hidden_state = self.next_state_head(features)
        next_hidden_state = self._normalize(next_hidden_state)

        reward = self.reward_head(features).squeeze(-1)

        return next_hidden_state, reward

    def _normalize(self, x: Tensor) -> Tensor:
        """Min-max normalize to [0, 1]."""
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        range_val = x_max - x_min
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        return (x - x_min) / range_val


class PredictionNetwork(nn.Module):
    """Predicts policy and value from hidden state.

    f(hidden_state) -> (policy, value)
    """

    def __init__(
        self,
        hidden_size: int,
        num_actions: int = 4,
        hidden_layers: List[int] = [256],
    ):
        """Initialize prediction network.

        Args:
            hidden_size: Size of hidden state
            num_actions: Number of possible actions
            hidden_layers: Hidden layer sizes
        """
        super().__init__()

        layers = []
        in_features = hidden_size

        for layer_size in hidden_layers:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.ReLU())
            in_features = layer_size

        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.policy_head = nn.Linear(in_features, num_actions)
        self.value_head = nn.Linear(in_features, 1)

    def forward(self, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict policy and value.

        Args:
            hidden_state: (N, hidden_size) hidden state

        Returns:
            Tuple of:
            - policy_logits: (N, 4) unnormalized action scores
            - value: (N,) state value estimates
        """
        features = self.trunk(hidden_state)

        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        return policy_logits, value

    def get_policy(
        self,
        hidden_state: Tensor,
        valid_mask: Tensor
    ) -> Tensor:
        """Get masked policy probabilities.

        Args:
            hidden_state: (N, hidden_size)
            valid_mask: (N, 4) valid action mask

        Returns:
            (N, 4) action probabilities
        """
        logits, _ = self.forward(hidden_state)

        # Mask invalid actions
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = float('-inf')

        return F.softmax(masked_logits, dim=1)


class MuZeroNetworks(nn.Module):
    """Container for all MuZero networks.

    Combines representation, dynamics, and prediction networks
    into a single module for easier training and checkpointing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        representation_layers: List[int] = [256],
        dynamics_layers: List[int] = [256],
        prediction_layers: List[int] = [256],
        num_actions: int = 4,
    ):
        """Initialize all MuZero networks.

        Args:
            input_size: Size of flattened input observation
            hidden_size: Size of hidden state
            representation_layers: Layers for representation network
            dynamics_layers: Layers for dynamics network
            prediction_layers: Layers for prediction network
            num_actions: Number of possible actions
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_actions = num_actions

        self.representation = RepresentationNetwork(
            input_size, hidden_size, representation_layers
        )
        self.dynamics = DynamicsNetwork(
            hidden_size, num_actions, dynamics_layers
        )
        self.prediction = PredictionNetwork(
            hidden_size, num_actions, prediction_layers
        )

    def initial_inference(
        self,
        observation: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Initial inference from observation.

        Args:
            observation: (N, 16, 17) game state
            valid_mask: (N, 4) valid actions mask

        Returns:
            Tuple of:
            - hidden_state: (N, hidden_size)
            - policy: (N, 4) action probabilities
            - value: (N,) value estimates
        """
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)

        # Mask invalid actions
        masked_logits = policy_logits.clone()
        masked_logits[~valid_mask] = float('-inf')
        policy = F.softmax(masked_logits, dim=1)

        return hidden_state, policy, value

    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Recurrent inference using dynamics model.

        Args:
            hidden_state: (N, hidden_size) current hidden state
            action: (N,) action to take
            valid_mask: (N, 4) valid actions mask (for next state)

        Returns:
            Tuple of:
            - next_hidden_state: (N, hidden_size)
            - reward: (N,) predicted reward
            - policy: (N, 4) action probabilities for next state
            - value: (N,) value estimates for next state
        """
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)

        # Mask invalid actions
        masked_logits = policy_logits.clone()
        masked_logits[~valid_mask] = float('-inf')
        policy = F.softmax(masked_logits, dim=1)

        return next_hidden_state, reward, policy, value
