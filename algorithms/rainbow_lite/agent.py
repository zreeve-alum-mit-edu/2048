"""
Rainbow-lite Agent.

Combines the best techniques from DQN improvements:
- Double DQN: Reduces overestimation bias
- Dueling architecture: Separates value and advantage
- Prioritized Experience Replay: Samples important transitions
- n-step returns: Faster value propagation

Excluded from full Rainbow (for simplicity):
- C51 (distributional RL)
- NoisyNet (uses epsilon-greedy instead)

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

from algorithms.rainbow_lite.model import DuelingDQNNetwork
from algorithms.rainbow_lite.priority_nstep_buffer import PrioritizedNStepBuffer
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class RainbowLiteAgent:
    """Rainbow-lite Agent for playing 2048.

    Combines:
    - Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean(A))
    - Double DQN target: Q_target(s', argmax_a Q_policy(s', a))
    - Prioritized replay: P(i) ~ |TD_error_i|^alpha
    - n-step returns: R_n + gamma^n * Q(s_n, a*)
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
        # n-step parameters
        n_steps: int = 3,
        # PER parameters
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
    ):
        """Initialize Rainbow-lite agent.

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
            per_alpha: Prioritization exponent
            per_beta_start: Initial importance sampling correction
            per_beta_frames: Frames to anneal beta to 1.0
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

        input_size = self.representation.output_shape()[0]

        # Dueling networks
        self.policy_net = DuelingDQNNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        self.target_net = DuelingDQNNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        # Prioritized n-step replay buffer
        self.replay_buffer = PrioritizedNStepBuffer(
            capacity=buffer_capacity,
            device=device,
            n_steps=n_steps,
            gamma=gamma,
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

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step.

        Combines all Rainbow-lite components:
        - Prioritized sampling with importance weights
        - n-step returns
        - Double DQN action selection/evaluation
        - (Implicit) Dueling architecture in network

        Returns:
            Dict with training metrics or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.buffer_min_size):
            return None

        # Sample prioritized batch with n-step returns
        (states, actions, nstep_returns, next_states, dones,
         valid_masks, actual_n, indices, weights) = self.replay_buffer.sample(self.batch_size)

        # Transform states
        repr_states = self.representation(states)
        with torch.no_grad():
            repr_next_states = self.representation(next_states)

        # Current Q values (through Dueling architecture)
        current_q_values = self.policy_net(repr_states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN + n-step target
        with torch.no_grad():
            # Policy net selects best action
            next_policy_q = self.policy_net.get_action_values(
                repr_next_states, valid_masks
            )
            best_actions = next_policy_q.argmax(dim=1)

            # Target net evaluates
            next_target_q = self.target_net(repr_next_states)
            next_q_values = next_target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            next_q_values = torch.where(dones, torch.zeros_like(next_q_values), next_q_values)

            # n-step TD target
            gamma_n = self.gamma ** actual_n.float()
            target_q = nstep_returns + gamma_n * next_q_values

        # TD errors for priority update
        td_errors = target_q - current_q

        # Weighted loss (importance sampling correction)
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
            "hidden_layers": [layer.out_features for layer in self.policy_net.trunk
                             if isinstance(layer, torch.nn.Linear)],
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
