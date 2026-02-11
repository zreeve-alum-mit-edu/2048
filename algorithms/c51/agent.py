"""
C51 Agent.

Implements Categorical DQN training with:
- Distributional value learning over fixed support
- Cross-entropy loss between projected target and prediction
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

from algorithms.c51.model import C51Network
from algorithms.double_dqn.replay_buffer import ReplayBuffer
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class C51Agent:
    """C51 (Categorical DQN) Agent for playing 2048.

    Key features:
    - Learns distribution of returns, not just expected value
    - Uses categorical cross-entropy loss
    - Projects Bellman target onto fixed support
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
        n_atoms: int = 51,
        v_min: float = 0.0,
        v_max: float = 100000.0,
    ):
        """Initialize C51 agent.

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
            n_atoms: Number of atoms in distribution support
            v_min: Minimum value of support
            v_max: Maximum value of support
        """
        self.device = device
        self.gamma = gamma
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_frequency = target_update_frequency
        self.batch_size = batch_size
        self.buffer_min_size = buffer_min_size

        # Compute support
        self.support = torch.linspace(v_min, v_max, n_atoms, device=device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        input_size = self.representation.output_shape()[0]

        # Networks
        self.policy_net = C51Network(
            input_size=input_size,
            hidden_layers=hidden_layers,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
        ).to(device)

        self.target_net = C51Network(
            input_size=input_size,
            hidden_layers=hidden_layers,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
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

    def _project_distribution(
        self,
        next_probs: Tensor,
        rewards: Tensor,
        dones: Tensor
    ) -> Tensor:
        """Project the Bellman target distribution onto the support.

        For each atom z_j of the target, compute:
        Tz_j = r + gamma * z_j (clipped to [v_min, v_max])

        Then distribute probability mass to neighboring atoms.

        Args:
            next_probs: (batch, n_atoms) target distribution for best action
            rewards: (batch,) immediate rewards
            dones: (batch,) done flags

        Returns:
            (batch, n_atoms) projected distribution
        """
        batch_size = rewards.size(0)

        # Compute Tz (shifted support)
        # For done states, Tz = r; for non-done, Tz = r + gamma * z
        rewards = rewards.unsqueeze(1)  # (batch, 1)
        dones = dones.unsqueeze(1).float()  # (batch, 1)
        support = self.support.unsqueeze(0)  # (1, n_atoms)

        Tz = rewards + (1 - dones) * self.gamma * support  # (batch, n_atoms)

        # Clip to support range
        Tz = Tz.clamp(self.v_min, self.v_max)

        # Compute projection indices
        b = (Tz - self.v_min) / self.delta_z  # (batch, n_atoms)
        lower = b.floor().long()
        upper = b.ceil().long()

        # Handle edge case where b is exactly an integer
        lower = lower.clamp(0, self.n_atoms - 1)
        upper = upper.clamp(0, self.n_atoms - 1)

        # Distribute probability mass
        projected = torch.zeros(batch_size, self.n_atoms, device=self.device)

        # Upper and lower interpolation weights
        upper_weight = b - lower.float()
        lower_weight = 1 - upper_weight

        # Scatter probability mass to lower and upper atoms
        # Using scatter_add for vectorized operation (DEC-0039)
        for atom_idx in range(self.n_atoms):
            # Mass from atom_idx goes to lower and upper neighbors
            mass = next_probs[:, atom_idx]  # (batch,)
            l_idx = lower[:, atom_idx]  # (batch,)
            u_idx = upper[:, atom_idx]  # (batch,)
            l_weight = lower_weight[:, atom_idx]  # (batch,)
            u_weight = upper_weight[:, atom_idx]  # (batch,)

            projected.scatter_add_(
                1,
                l_idx.unsqueeze(1),
                (mass * l_weight).unsqueeze(1)
            )
            projected.scatter_add_(
                1,
                u_idx.unsqueeze(1),
                (mass * u_weight).unsqueeze(1)
            )

        return projected

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step.

        Uses cross-entropy loss between predicted and projected target distributions.

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

        # Get current distributions
        current_probs = self.policy_net(repr_states)  # (batch, 4, n_atoms)
        # Select distributions for taken actions
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)
        current_dist = current_probs.gather(1, actions_expanded).squeeze(1)  # (batch, n_atoms)

        # Compute target distribution
        with torch.no_grad():
            # Get next action from policy network (Double DQN style)
            next_q_values = self.policy_net.get_action_values(
                repr_next_states, valid_masks
            )
            next_actions = next_q_values.argmax(dim=1)  # (batch,)

            # Get target distribution for selected actions
            target_probs = self.target_net(repr_next_states)  # (batch, 4, n_atoms)
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)
            next_dist = target_probs.gather(1, next_actions_expanded).squeeze(1)  # (batch, n_atoms)

            # Project target distribution
            projected_dist = self._project_distribution(next_dist, rewards, dones)

        # Cross-entropy loss (KL divergence up to constant)
        # -sum(target * log(predicted))
        # Add small epsilon for numerical stability
        log_current = torch.log(current_dist + 1e-8)
        loss = -(projected_dist * log_current).sum(dim=1).mean()

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
            "n_atoms": self.n_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
