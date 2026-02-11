"""
Expected SARSA Agent.

Implements Expected SARSA which uses the expected Q-value under the
policy for bootstrapping instead of the sampled next action:
Q(s,a) += alpha * (r + gamma * E_pi[Q(s',a')] - Q(s,a))

where E_pi[Q(s',a')] = sum_a' pi(a'|s') * Q(s',a')

This reduces variance compared to SARSA by eliminating the randomness
from action selection in the target.

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
Per DEC-0039: Vectorized tensor operations
"""

from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from algorithms.expected_sarsa.model import ExpectedSARSANetwork
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class ExpectedSARSAAgent:
    """Expected SARSA Agent for playing 2048.

    Uses expected Q-value under epsilon-greedy policy:
    E[Q(s',a')] = (1-eps) * max_a Q(s',a) + eps/|A| * sum_a Q(s',a)

    This is equivalent to weighting Q-values by action probabilities.
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
        """Initialize Expected SARSA agent.

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

        input_size = self.representation.output_shape()[0]

        # Q-Network
        self.network = ExpectedSARSANetwork(
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

    def _compute_epsilon(self) -> float:
        """Compute current epsilon based on linear decay schedule."""
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_end
        fraction = self.step_count / self.epsilon_decay_steps
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _compute_policy_probs(
        self,
        q_values: Tensor,
        valid_mask: Tensor,
        epsilon: float
    ) -> Tensor:
        """Compute epsilon-greedy policy probabilities.

        Args:
            q_values: (N, 4) Q-values
            valid_mask: (N, 4) valid action mask
            epsilon: Current epsilon

        Returns:
            (N, 4) action probabilities under epsilon-greedy policy
        """
        batch_size = q_values.size(0)
        num_actions = q_values.size(1)

        # Mask invalid Q-values
        masked_q = q_values.clone()
        masked_q[~valid_mask] = float('-inf')

        # Find greedy action
        greedy_actions = masked_q.argmax(dim=1)

        # Count valid actions per state (for uniform random part)
        num_valid = valid_mask.float().sum(dim=1, keepdim=True)  # (N, 1)
        num_valid = torch.clamp(num_valid, min=1)  # Avoid division by zero

        # Initialize probabilities
        probs = torch.zeros(batch_size, num_actions, device=self.device)

        # Epsilon part: uniform over valid actions
        probs[valid_mask] = epsilon / num_valid.expand(-1, num_actions)[valid_mask]

        # Add (1-epsilon) to greedy action
        greedy_one_hot = F.one_hot(greedy_actions, num_actions).float()
        probs = probs + (1 - epsilon) * greedy_one_hot

        return probs

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

        with torch.no_grad():
            repr_state = self.representation(state)

        if training:
            self.epsilon = self._compute_epsilon()

            random_mask = torch.rand(batch_size, device=self.device) < self.epsilon

            with torch.no_grad():
                q_values = self.network.get_action_values(repr_state, valid_mask)
                greedy_actions = q_values.argmax(dim=1)

            # Random valid actions
            probs = valid_mask.float()
            row_sums = probs.sum(dim=1, keepdim=True)
            no_valid = (row_sums == 0).squeeze(1)
            if no_valid.any():
                probs[no_valid, 0] = 1.0
                row_sums = probs.sum(dim=1, keepdim=True)
            probs = probs / row_sums
            random_actions = torch.multinomial(probs, 1).squeeze(1)

            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
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
        done: Tensor,
        valid_mask: Tensor,
        next_valid_mask: Tensor
    ) -> Dict[str, float]:
        """Perform one Expected SARSA update.

        Expected SARSA: Q(s,a) += alpha * (r + gamma * E[Q(s',a')] - Q(s,a))
        where E[Q(s',a')] = sum_a' pi(a'|s') * Q(s',a')

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards received
            next_state: (N, 16, 17) next states
            done: (N,) done flags
            valid_mask: (N, 4) valid masks for current states
            next_valid_mask: (N, 4) valid masks for next states

        Returns:
            Dict with training metrics
        """
        # Normalize rewards
        if reward.std() > 0:
            reward_norm = (reward - reward.mean()) / (reward.std() + 1e-8)
        else:
            reward_norm = reward

        # Transform states
        repr_state = self.representation(state)
        with torch.no_grad():
            repr_next_state = self.representation(next_state)

        # Current Q-values
        q_values = self.network(repr_state)
        current_q = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Expected next Q-value under epsilon-greedy policy
        with torch.no_grad():
            next_q_values = self.network(repr_next_state)

            # Compute policy probabilities
            probs = self._compute_policy_probs(next_q_values, next_valid_mask, self.epsilon)

            # Expected Q = sum_a pi(a|s') * Q(s',a)
            # Mask invalid actions' Q-values to 0 for the expectation
            next_q_masked = next_q_values.clone()
            next_q_masked[~next_valid_mask] = 0

            expected_q = (probs * next_q_masked).sum(dim=1)

            # Zero for terminal states
            expected_q = torch.where(done, torch.zeros_like(expected_q), expected_q)

            # TD target
            target_q = reward_norm + self.gamma * expected_q

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
            "expected_q_mean": expected_q.mean().item(),
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
