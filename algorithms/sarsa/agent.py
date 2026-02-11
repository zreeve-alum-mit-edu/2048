"""
SARSA (State-Action-Reward-State-Action) Agent.

Implements on-policy TD control:
Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))

Key difference from Q-learning (DQN):
- DQN uses max_a Q(s',a) for target (off-policy)
- SARSA uses Q(s',a') where a' is the actual next action (on-policy)

This makes SARSA more conservative as it accounts for the exploration
policy when learning.

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
Per DEC-0039: Vectorized tensor operations
"""

from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from algorithms.sarsa.model import SARSANetwork
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class SARSAAgent:
    """SARSA Agent for playing 2048.

    On-policy TD control algorithm that uses the actual next action
    for bootstrapping, making it account for exploration.

    Key characteristics:
    - No replay buffer (on-policy)
    - Uses epsilon-greedy for exploration
    - Updates based on actual transitions taken
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: List[int] = [256, 256],
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
    ):
        """Initialize SARSA agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon after decay
            epsilon_decay_steps: Steps for linear epsilon decay
        """
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        # Get input size from representation
        input_size = self.representation.output_shape()[0]

        # Q-Network
        self.network = SARSANetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate
        )

        # Training state
        self.step_count = 0
        self.epsilon = epsilon_start
        self.hidden_layers = hidden_layers

        # Previous state/action for SARSA update
        self._prev_state: Optional[Tensor] = None
        self._prev_action: Optional[Tensor] = None
        self._prev_valid_mask: Optional[Tensor] = None

    def _compute_epsilon(self) -> float:
        """Compute current epsilon based on linear decay schedule.

        Returns:
            Current epsilon value
        """
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_end

        fraction = self.step_count / self.epsilon_decay_steps
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(
        self,
        state: Tensor,
        valid_mask: Tensor,
        training: bool = True
    ) -> Tensor:
        """Select actions for a batch of states.

        Args:
            state: (N, 16, 17) current board states
            valid_mask: (N, 4) valid action masks
            training: If True, use epsilon-greedy; if False, greedy

        Returns:
            (N,) selected actions
        """
        batch_size = state.size(0)

        # Transform state through representation
        with torch.no_grad():
            repr_state = self.representation(state)

        if training:
            self.epsilon = self._compute_epsilon()

            # Epsilon-greedy action selection
            random_mask = torch.rand(batch_size, device=self.device) < self.epsilon

            # Get greedy actions
            with torch.no_grad():
                q_values = self.network.get_action_values(repr_state, valid_mask)
                greedy_actions = q_values.argmax(dim=1)

            # Random valid actions (vectorized per DEC-0039)
            probs = valid_mask.float()
            row_sums = probs.sum(dim=1, keepdim=True)
            no_valid = (row_sums == 0).squeeze(1)
            if no_valid.any():
                probs[no_valid, 0] = 1.0
                row_sums = probs.sum(dim=1, keepdim=True)
            probs = probs / row_sums
            random_actions = torch.multinomial(probs, 1).squeeze(1)

            # Combine greedy and random actions
            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
            # Pure greedy
            with torch.no_grad():
                q_values = self.network.get_action_values(repr_state, valid_mask)
                actions = q_values.argmax(dim=1)

        return actions

    def train_step(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        next_action: Tensor,
        done: Tensor,
        valid_mask: Tensor,
        next_valid_mask: Tensor
    ) -> Dict[str, float]:
        """Perform one SARSA update.

        SARSA update: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards received
            next_state: (N, 16, 17) next states
            next_action: (N,) next actions (for SARSA)
            done: (N,) done flags
            valid_mask: (N, 4) valid masks for current states
            next_valid_mask: (N, 4) valid masks for next states

        Returns:
            Dict with training metrics
        """
        # Normalize rewards for stability
        if reward.std() > 0:
            reward_norm = (reward - reward.mean()) / (reward.std() + 1e-8)
        else:
            reward_norm = reward

        # Transform states through representation
        repr_state = self.representation(state)
        with torch.no_grad():
            repr_next_state = self.representation(next_state)

        # Current Q-values
        q_values = self.network(repr_state)
        current_q = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Next Q-values using actual next action (SARSA key difference)
        with torch.no_grad():
            next_q_values = self.network(repr_next_state)
            next_q = next_q_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            # Zero for terminal states
            next_q = torch.where(done, torch.zeros_like(next_q), next_q)

            # TD target
            target_q = reward_norm + self.gamma * next_q

        # Loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "epsilon": self.epsilon,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            "hidden_layers": self.hidden_layers,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
