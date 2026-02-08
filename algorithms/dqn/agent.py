"""
DQN Agent.

Implements the DQN algorithm with:
- Epsilon-greedy exploration with linear decay (DEC-0035)
- Mask-based action selection (DEC-0034)
- Hard target network updates (DEC-0036)
- Experience replay respecting episode boundaries (DEC-0003)
- Support for different representations (DEC-0037/Milestone 5)
"""

import copy
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from algorithms.dqn.model import DQNNetwork
from algorithms.dqn.replay_buffer import ReplayBuffer
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class DQNAgent:
    """DQN Agent for playing 2048.

    Key decisions implemented:
    - DEC-0033: Uses merge_reward only (for basic milestone)
    - DEC-0034: Mask-based action selection (invalid Q-values set to -inf)
    - DEC-0035: Linear epsilon decay over 100k steps
    - DEC-0036: Hard target network update every N steps
    - DEC-0037: Supports multiple representations via representation parameter
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
    ):
        """Initialize DQN agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
                           If None, uses OneHotRepresentation (backward compatible).
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
        # Default to OneHotRepresentation for backward compatibility
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        # Get input size from representation
        input_size = self.representation.output_shape()[0]

        # Networks with dynamic input size
        self.policy_net = DQNNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        self.target_net = DQNNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, device)

        # Training state
        self.step_count = 0
        self.epsilon = epsilon_start

    def _compute_epsilon(self) -> float:
        """Compute current epsilon based on linear decay schedule.

        Per DEC-0035: Linear decay from epsilon_start to epsilon_end
        over epsilon_decay_steps.

        Returns:
            Current epsilon value
        """
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_end

        # Linear interpolation
        fraction = self.step_count / self.epsilon_decay_steps
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(
        self,
        state: Tensor,
        valid_mask: Tensor,
        training: bool = True
    ) -> Tensor:
        """Select actions for a batch of states.

        Per DEC-0034: Use mask-based action selection.
        Invalid Q-values are set to -inf, ensuring only valid actions
        are selected.

        Args:
            state: (N, 16, 17) current board states
            valid_mask: (N, 4) valid action masks
            training: If True, use epsilon-greedy; if False, greedy only

        Returns:
            (N,) selected actions
        """
        batch_size = state.size(0)

        # Transform state through representation (DEC-0037)
        with torch.no_grad():
            repr_state = self.representation(state)

        if training:
            self.epsilon = self._compute_epsilon()

            # Epsilon-greedy action selection
            random_mask = torch.rand(batch_size, device=self.device) < self.epsilon

            # Get greedy actions for non-random selections
            with torch.no_grad():
                q_values = self.policy_net.get_action_values(repr_state, valid_mask)
                greedy_actions = q_values.argmax(dim=1)

            # Random valid actions for exploration (vectorized per DEC-0039)
            # Use multinomial sampling with valid_mask as probability distribution
            probs = valid_mask.float()
            probs = probs / probs.sum(dim=1, keepdim=True)
            random_actions = torch.multinomial(probs, 1).squeeze(1)

            # Combine greedy and random actions
            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
            # Pure greedy action selection
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
        """Store a batch of transitions in the replay buffer.

        Per DEC-0003: done=True means next_state is terminal.
        No cross-episode transitions are stored.

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards (merge_reward per DEC-0033)
            next_state: (N, 16, 17) next states
            done: (N,) episode termination flags
            valid_mask: (N, 4) valid actions for next states
        """
        self.replay_buffer.push(
            state, action, reward, next_state, done, valid_mask
        )

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step.

        Returns:
            Dict with training metrics (loss, q_values) or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.buffer_min_size):
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, valid_masks = \
            self.replay_buffer.sample(self.batch_size)

        # Transform states through representation (DEC-0037)
        repr_states = self.representation(states)
        with torch.no_grad():
            repr_next_states = self.representation(next_states)

        # Compute current Q values
        current_q_values = self.policy_net(repr_states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            # Get Q-values from target network with masking
            next_q_values = self.target_net.get_action_values(
                repr_next_states, valid_masks
            )
            # Max Q-value for non-terminal states
            next_q_max = next_q_values.max(dim=1)[0]
            # For terminal states, next Q is 0
            next_q_max = torch.where(dones, torch.zeros_like(next_q_max), next_q_max)
            # TD target
            target_q = rewards + self.gamma * next_q_max

        # Compute loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update step count
        self.step_count += 1

        # Hard target network update (DEC-0036)
        if self.step_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
            "epsilon": self.epsilon,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            # Save architecture info for loading
            "hidden_layers": [layer.out_features for layer in self.policy_net.network
                             if isinstance(layer, torch.nn.Linear)][:-1],  # Exclude output layer
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
