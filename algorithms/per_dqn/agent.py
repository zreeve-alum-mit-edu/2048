"""
PER DQN (Prioritized Experience Replay DQN) Agent.

Implements DQN with prioritized experience replay which samples
transitions proportionally to their TD error, allowing the agent
to focus on more surprising/informative experiences.

Key features:
- Proportional prioritization: P(i) ~ |TD_error_i|^alpha
- Importance sampling weights to correct bias
- Beta annealing from beta_start to 1.0

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from algorithms.per_dqn.model import DoubleDQNNetwork
from algorithms.per_dqn.priority_buffer import PrioritizedReplayBuffer
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class PERDQNAgent:
    """PER DQN Agent for playing 2048.

    Combines Double DQN target computation with prioritized replay:
    - Samples transitions with probability proportional to TD error
    - Uses importance sampling weights in loss computation
    - Updates priorities after each training step
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: list = [256, 256],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        target_update_frequency: int = 1000,
        buffer_capacity: int = 100000,
        buffer_min_size: int = 1000,
        batch_size: int = 64,
        # PER specific parameters
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
    ):
        """Initialize PER DQN agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon after decay
            epsilon_decay_steps: Steps for linear epsilon decay
            target_update_frequency: Steps between target network updates
            buffer_capacity: Replay buffer capacity
            buffer_min_size: Minimum buffer size before training
            batch_size: Batch size for training
            per_alpha: Prioritization exponent (0=uniform, 1=full)
            per_beta_start: Initial importance sampling correction
            per_beta_frames: Frames to anneal beta to 1.0
        """
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_frequency = target_update_frequency
        self.batch_size = batch_size
        self.buffer_min_size = buffer_min_size

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        # Get input size from representation
        input_size = self.representation.output_shape()[0]

        # Networks
        self.policy_net = DoubleDQNNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        self.target_net = DoubleDQNNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            device=device,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )

        # Training state
        self.step_count = 0
        self.epsilon = epsilon_start

    def _compute_epsilon(self) -> float:
        """Compute current epsilon based on linear decay."""
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
            training: If True, use epsilon-greedy; if False, greedy only

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
                q_values = self.policy_net.get_action_values(repr_state, valid_mask)
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
                q_values = self.policy_net.get_action_values(repr_state, valid_mask)
                actions = q_values.argmax(dim=1)

        return actions

    def store_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        valid_mask: Tensor
    ) -> None:
        """Store a batch of transitions in the prioritized buffer."""
        self.replay_buffer.push(
            state, action, reward, next_state, done, valid_mask
        )

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step with prioritized replay.

        Key differences from standard DQN:
        1. Sample transitions based on priority
        2. Weight loss by importance sampling weights
        3. Update priorities after computing TD errors

        Returns:
            Dict with training metrics or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.buffer_min_size):
            return None

        # Sample prioritized batch (includes weights and indices)
        states, actions, rewards, next_states, dones, valid_masks, indices, weights = \
            self.replay_buffer.sample(self.batch_size)

        # Transform states
        repr_states = self.representation(states)
        with torch.no_grad():
            repr_next_states = self.representation(next_states)

        # Current Q values
        current_q_values = self.policy_net(repr_states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_policy_q = self.policy_net.get_action_values(
                repr_next_states, valid_masks
            )
            best_actions = next_policy_q.argmax(dim=1)

            next_target_q = self.target_net(repr_next_states)
            next_q_values = next_target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            next_q_values = torch.where(dones, torch.zeros_like(next_q_values), next_q_values)

            target_q = rewards + self.gamma * next_q_values

        # Compute TD errors for priority update
        td_errors = target_q - current_q

        # Weighted loss (importance sampling correction)
        # L = w_i * (target - Q)^2
        elementwise_loss = (current_q - target_q) ** 2
        weighted_loss = (weights * elementwise_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach())

        self.step_count += 1

        # Hard target update
        if self.step_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            "loss": weighted_loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
            "epsilon": self.epsilon,
            "td_error_mean": td_errors.abs().mean().item(),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            "hidden_layers": [layer.out_features for layer in self.policy_net.network
                             if isinstance(layer, torch.nn.Linear)][:-1],
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
