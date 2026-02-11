"""
n-step DQN Agent.

Implements DQN with multi-step returns for faster value propagation.
Instead of 1-step TD target: r + gamma * Q(s', a*)
Uses n-step TD target: R_n + gamma^n * Q(s_n, a*)

where R_n = r_0 + gamma*r_1 + ... + gamma^{n-1}*r_{n-1}

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

from algorithms.nstep_dqn.model import DoubleDQNNetwork
from algorithms.nstep_dqn.nstep_buffer import NStepReplayBuffer
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class NStepDQNAgent:
    """n-step DQN Agent for playing 2048.

    Combines Double DQN target computation with n-step returns:
    - n-step return: R_n = sum_{i=0}^{n-1} gamma^i * r_i
    - Target: R_n + gamma^n * Q_target(s_n, argmax_a Q_policy(s_n, a))
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
        n_steps: int = 3,  # Multi-step parameter
    ):
        """Initialize n-step DQN agent.

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
            n_steps: Number of steps for multi-step returns
        """
        self.device = device
        self.gamma = gamma
        self.n_steps = n_steps
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

        # n-step replay buffer
        self.replay_buffer = NStepReplayBuffer(
            capacity=buffer_capacity,
            device=device,
            n_steps=n_steps,
            gamma=gamma,
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
        """Store a batch of transitions in the n-step buffer."""
        self.replay_buffer.push(
            state, action, reward, next_state, done, valid_mask
        )

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step with n-step returns.

        Key difference: Uses pre-computed n-step returns and gamma^n discount.

        Returns:
            Dict with training metrics or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.buffer_min_size):
            return None

        # Sample includes n-step returns and actual_n
        states, actions, nstep_returns, next_states, dones, valid_masks, actual_n = \
            self.replay_buffer.sample(self.batch_size)

        # Transform states
        repr_states = self.representation(states)
        with torch.no_grad():
            repr_next_states = self.representation(next_states)

        # Current Q values
        current_q_values = self.policy_net(repr_states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # n-step Double DQN target
        with torch.no_grad():
            # Select best action using policy net
            next_policy_q = self.policy_net.get_action_values(
                repr_next_states, valid_masks
            )
            best_actions = next_policy_q.argmax(dim=1)

            # Evaluate using target net
            next_target_q = self.target_net(repr_next_states)
            next_q_values = next_target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # Zero for terminal states
            next_q_values = torch.where(dones, torch.zeros_like(next_q_values), next_q_values)

            # n-step TD target: R_n + gamma^n * Q(s_n, a*)
            # Note: gamma^n depends on actual_n (may be < n_steps if episode ended)
            gamma_n = self.gamma ** actual_n.float()
            target_q = nstep_returns + gamma_n * next_q_values

        # Loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1

        # Hard target update
        if self.step_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
            "epsilon": self.epsilon,
            "avg_n_steps": actual_n.float().mean().item(),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            "n_steps": self.n_steps,
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
