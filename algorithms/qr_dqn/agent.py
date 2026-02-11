"""
QR-DQN Agent.

Implements Quantile Regression DQN training with:
- Fixed quantile fractions, learned quantile values
- Huber quantile regression loss
- Experience replay buffer
- Target network for stable training

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

from algorithms.qr_dqn.model import QRNetwork
from algorithms.double_dqn.replay_buffer import ReplayBuffer
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class QRDQNAgent:
    """QR-DQN (Quantile Regression DQN) Agent for playing 2048.

    Key features:
    - Learns quantile values instead of expected value
    - Uses Huber quantile regression loss
    - No need to specify value bounds (unlike C51)
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
        n_quantiles: int = 200,
        kappa: float = 1.0,  # Huber loss threshold
    ):
        """Initialize QR-DQN agent.

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
            n_quantiles: Number of quantiles to estimate
            kappa: Threshold for Huber loss
        """
        self.device = device
        self.gamma = gamma
        self.n_quantiles = n_quantiles
        self.kappa = kappa
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_frequency = target_update_frequency
        self.batch_size = batch_size
        self.buffer_min_size = buffer_min_size

        # Quantile midpoints: tau_i = (i + 0.5) / N
        self.taus = (torch.arange(n_quantiles, device=device, dtype=torch.float32) + 0.5) / n_quantiles

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        input_size = self.representation.output_shape()[0]

        # Networks
        self.policy_net = QRNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers,
            n_quantiles=n_quantiles,
        ).to(device)

        self.target_net = QRNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers,
            n_quantiles=n_quantiles,
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        # Replay buffer (reuse from Double DQN)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            device=device,
        )

        # Training state
        self.step_count = 0
        self.epsilon = epsilon_start
        self.hidden_layers = hidden_layers

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

            # Random action selection over valid actions
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
        """Store a batch of transitions."""
        self.replay_buffer.push(
            state, action, reward, next_state, done, valid_mask
        )

    def _huber_quantile_loss(
        self,
        quantiles: Tensor,
        target_quantiles: Tensor,
        taus: Tensor
    ) -> Tensor:
        """Compute Huber quantile regression loss.

        Args:
            quantiles: (batch, n_quantiles) predicted quantile values
            target_quantiles: (batch, n_quantiles) target quantile values
            taus: (n_quantiles,) quantile fractions

        Returns:
            Scalar loss
        """
        # Expand for pairwise computation
        # quantiles: (batch, n_quantiles, 1)
        # target_quantiles: (batch, 1, n_quantiles)
        quantiles = quantiles.unsqueeze(2)  # (batch, N, 1)
        target_quantiles = target_quantiles.unsqueeze(1)  # (batch, 1, N)

        # TD errors: (batch, N, N)
        td_errors = target_quantiles - quantiles

        # Huber loss
        huber_loss = torch.where(
            td_errors.abs() <= self.kappa,
            0.5 * td_errors ** 2,
            self.kappa * (td_errors.abs() - 0.5 * self.kappa)
        )

        # Quantile regression weights
        # tau * |u| if u < 0, (1 - tau) * |u| if u >= 0
        # Using indicator: tau - I(u < 0)
        taus = taus.view(1, -1, 1)  # (1, N, 1)
        indicator = (td_errors < 0).float()
        quantile_weights = torch.abs(taus - indicator)

        # Weighted loss
        loss = (quantile_weights * huber_loss).sum(dim=2).mean(dim=1)

        return loss.mean()

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step.

        Uses Huber quantile regression loss.

        Returns:
            Dict with training metrics or None if buffer not ready
        """
        if len(self.replay_buffer) < self.buffer_min_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones, valid_masks = \
            self.replay_buffer.sample(self.batch_size)

        # Transform states
        repr_states = self.representation(states)
        with torch.no_grad():
            repr_next_states = self.representation(next_states)

        # Get current quantile values
        current_quantiles = self.policy_net(repr_states)  # (batch, 4, n_quantiles)
        # Select quantiles for taken actions
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_quantiles)
        current_q_quantiles = current_quantiles.gather(1, actions_expanded).squeeze(1)  # (batch, n_quantiles)

        # Compute target quantiles
        with torch.no_grad():
            # Get next action from policy network (Double DQN style)
            next_q_values = self.policy_net.get_action_values(
                repr_next_states, valid_masks
            )
            next_actions = next_q_values.argmax(dim=1)  # (batch,)

            # Get target quantiles for selected actions
            target_quantiles = self.target_net(repr_next_states)  # (batch, 4, n_quantiles)
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_quantiles)
            next_q_quantiles = target_quantiles.gather(1, next_actions_expanded).squeeze(1)  # (batch, n_quantiles)

            # Compute target: r + gamma * quantile_value (for non-done)
            rewards_expanded = rewards.unsqueeze(1)  # (batch, 1)
            dones_expanded = dones.unsqueeze(1).float()  # (batch, 1)
            target_q_quantiles = rewards_expanded + (1 - dones_expanded) * self.gamma * next_q_quantiles

        # Compute loss
        loss = self._huber_quantile_loss(current_q_quantiles, target_q_quantiles, self.taus)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1

        # Hard target update
        if self.step_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Compute Q-values for logging
        with torch.no_grad():
            q_values = self.policy_net.get_q_values(repr_states)
            q_mean = q_values.mean().item()

        return {
            "loss": loss.item(),
            "q_mean": q_mean,
            "epsilon": self.epsilon,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            "hidden_layers": self.hidden_layers,
            "n_quantiles": self.n_quantiles,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
