"""
ACER (Actor-Critic with Experience Replay) Agent.

Implements ACER algorithm which combines on-policy actor-critic with
off-policy experience replay using importance sampling correction.

Key ACER components:
- Experience replay for sample efficiency
- Importance sampling with truncation for stability
- Retrace(lambda) for off-policy correction
- Bias correction term

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
Per DEC-0039: Vectorized tensor operations
"""

from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from algorithms.acer.model import ACERNetwork
from algorithms.acer.replay_buffer import ACERReplayBuffer
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class ACERAgent:
    """ACER Agent for playing 2048.

    ACER combines actor-critic with experience replay by using
    importance sampling to correct for the difference between
    the current policy and the behavior policy that collected
    the experience.

    Key components:
    - On-policy updates (like A2C)
    - Off-policy updates from replay buffer with Retrace correction
    - Truncated importance sampling for variance reduction
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: List[int] = [256, 256],
        learning_rate: float = 0.0007,
        gamma: float = 0.99,
        n_steps: int = 5,
        c: float = 10.0,
        delta: float = 1.0,
        replay_ratio: int = 4,
        buffer_capacity: int = 5000,
        buffer_min_size: int = 100,
        batch_size: int = 16,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """Initialize ACER agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            n_steps: Number of steps for trajectories
            c: Importance sampling truncation constant
            delta: Trust region constraint (not used in basic version)
            replay_ratio: Number of replay updates per on-policy update
            buffer_capacity: Replay buffer capacity (in trajectories)
            buffer_min_size: Minimum trajectories before replay
            batch_size: Batch size for replay updates
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
        """
        self.device = device
        self.gamma = gamma
        self.n_steps = n_steps
        self.c = c  # Importance weight truncation
        self.replay_ratio = replay_ratio
        self.batch_size = batch_size
        self.buffer_min_size = buffer_min_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        # Get input size from representation
        input_size = self.representation.output_shape()[0]

        # Network (uses Q-values unlike A2C/A3C)
        self.network = ACERNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

        # Replay buffer will be initialized when we know n_games
        self.replay_buffer: Optional[ACERReplayBuffer] = None
        self.buffer_capacity = buffer_capacity
        self.n_games: Optional[int] = None

        # Rollout storage for on-policy collection
        self._states: List[Tensor] = []
        self._actions: List[Tensor] = []
        self._rewards: List[Tensor] = []
        self._dones: List[Tensor] = []
        self._valid_masks: List[Tensor] = []
        self._mu: List[Tensor] = []  # Behavior policy probs

        # Training state
        self.step_count = 0
        self.hidden_layers = hidden_layers

    def _init_replay_buffer(self, n_games: int) -> None:
        """Initialize replay buffer with known n_games.

        Args:
            n_games: Number of parallel games
        """
        self.n_games = n_games
        self.replay_buffer = ACERReplayBuffer(
            capacity=self.buffer_capacity,
            trajectory_length=self.n_steps,
            n_games=n_games,
            device=self.device
        )

    def select_action(
        self,
        state: Tensor,
        valid_mask: Tensor,
        training: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Select actions and return behavior policy probabilities.

        Args:
            state: (N, 16, 17) current board states
            valid_mask: (N, 4) valid action masks
            training: If True, sample; if False, greedy

        Returns:
            Tuple of:
            - actions: (N,) selected actions
            - mu: (N,) probability of selected action under behavior policy
        """
        if self.n_games is None:
            self._init_replay_buffer(state.size(0))

        # Transform state through representation
        with torch.no_grad():
            repr_state = self.representation(state)
            probs = self.network.get_action_probs(repr_state, valid_mask)

        if training:
            # Sample from the distribution
            probs_safe = probs.clone()
            row_sums = probs_safe.sum(dim=1, keepdim=True)
            no_valid = (row_sums == 0).squeeze(1)
            if no_valid.any():
                probs_safe[no_valid] = valid_mask[no_valid].float()
                probs_safe[no_valid] = probs_safe[no_valid] / probs_safe[no_valid].sum(dim=1, keepdim=True)

            actions = torch.multinomial(probs_safe, 1).squeeze(1)
            # Get probability of selected action
            mu = probs_safe.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            # Greedy action selection
            actions = probs.argmax(dim=1)
            mu = probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        return actions, mu

    def store_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        valid_mask: Tensor,
        mu: Tensor
    ) -> None:
        """Store a transition for trajectory collection.

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards
            done: (N,) episode termination flags
            valid_mask: (N, 4) valid actions for current states
            mu: (N,) behavior policy probability of action
        """
        self._states.append(state.clone())
        self._actions.append(action.clone())
        self._rewards.append(reward.clone())
        self._dones.append(done.clone())
        self._valid_masks.append(valid_mask.clone())
        self._mu.append(mu.clone())

    def _compute_retrace_targets(
        self,
        rewards: Tensor,
        dones: Tensor,
        q_values: Tensor,
        actions: Tensor,
        pi: Tensor,
        mu: Tensor,
        bootstrap_value: Tensor
    ) -> Tensor:
        """Compute Retrace targets for off-policy correction.

        Retrace(lambda) with lambda=1 (simplified):
        Q_ret = r + gamma * (rho * (Q_ret' - Q') + V')

        where rho = min(c, pi/mu) is the truncated importance weight.

        Args:
            rewards: (T, B) rewards
            dones: (T, B) done flags
            q_values: (T, B, 4) Q-values at each step
            actions: (T, B) actions taken
            pi: (T, B, 4) current policy probabilities
            mu: (T, B) behavior policy probabilities
            bootstrap_value: (B,) value for bootstrapping

        Returns:
            (T, B) Retrace Q-value targets
        """
        T, B = rewards.shape

        # Get Q-values of taken actions
        q_a = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (T, B)

        # Compute V = E_a[Q] under current policy
        v = (pi * q_values).sum(dim=-1)  # (T, B)

        # Compute importance weights: rho = min(c, pi(a)/mu(a))
        # Get pi(a) for actions taken
        pi_a = pi.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (T, B)
        rho = torch.clamp(pi_a / (mu + 1e-8), max=self.c)

        # Compute Retrace targets (backwards)
        q_ret = torch.zeros(T, B, device=self.device)
        q_ret_next = bootstrap_value

        for t in reversed(range(T)):
            # Reset on episode boundary
            q_ret_next = torch.where(dones[t], torch.zeros_like(q_ret_next), q_ret_next)

            # Retrace update
            q_ret[t] = rewards[t] + self.gamma * (
                rho[t] * (q_ret_next - q_a[t]) + v[t]
            )
            q_ret_next = q_ret[t]

        return q_ret

    def _update_from_trajectories(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        valid_masks: Tensor,
        mu: Tensor,
        next_state: Tensor,
        next_valid_mask: Tensor,
        on_policy: bool = True
    ) -> Dict[str, float]:
        """Perform update from trajectory data.

        Args:
            states: (T, B, 16, 17) states
            actions: (T, B) actions
            rewards: (T, B) rewards
            dones: (T, B) done flags
            valid_masks: (T, B, 4) valid action masks
            mu: (T, B) behavior policy probabilities
            next_state: (B, 16, 17) state for bootstrapping
            next_valid_mask: (B, 4) valid mask for bootstrap state
            on_policy: Whether this is on-policy update

        Returns:
            Dict with training metrics
        """
        T, B = actions.shape

        # Normalize rewards
        rewards_flat = rewards.view(-1)
        if rewards_flat.std() > 0:
            rewards_norm = (rewards - rewards_flat.mean()) / (rewards_flat.std() + 1e-8)
        else:
            rewards_norm = rewards

        # Transform states through representation
        flat_states = states.view(-1, 16, 17)
        repr_flat = self.representation(flat_states)
        repr_size = repr_flat.size(-1)
        repr_states = repr_flat.view(T, B, repr_size)

        # Get bootstrap value
        with torch.no_grad():
            repr_next = self.representation(next_state)
            bootstrap_v = self.network.get_value(repr_next, next_valid_mask)

        # Forward pass to get policy and Q-values
        flat_repr = repr_states.view(-1, repr_size)
        flat_masks = valid_masks.view(-1, 4)

        probs, log_probs, q_values = self.network.get_policy_and_q(flat_repr, flat_masks)

        probs = probs.view(T, B, 4)
        log_probs = log_probs.view(T, B, 4)
        q_values = q_values.view(T, B, 4)

        # Compute Retrace targets
        with torch.no_grad():
            q_ret = self._compute_retrace_targets(
                rewards_norm, dones, q_values.detach(),
                actions, probs.detach(), mu, bootstrap_v
            )

        # Get Q-values and log probs for taken actions
        q_a = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (T, B)
        log_prob_a = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (T, B)

        # Compute V = E[Q] under current policy
        v = (probs * q_values).sum(dim=-1)  # (T, B)

        # Advantage for policy gradient
        adv = q_ret - v.detach()

        # Importance weight for policy gradient
        pi_a = probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        rho = torch.clamp(pi_a / (mu + 1e-8), max=self.c)

        # Policy loss (importance-weighted)
        policy_loss = -(rho.detach() * adv * log_prob_a).mean()

        # Q-value loss
        q_loss = ((q_a - q_ret) ** 2).mean()

        # Entropy bonus
        entropy_per_state = -(probs * log_probs).sum(dim=-1)
        valid_entropy = entropy_per_state[entropy_per_state.isfinite()]
        entropy = valid_entropy.mean() if valid_entropy.numel() > 0 else torch.tensor(0.0, device=self.device)

        # Total loss
        loss = policy_loss + self.value_loss_coef * q_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients
        has_nan = any(
            param.grad is not None and torch.isnan(param.grad).any()
            for param in self.network.parameters()
        )

        if not has_nan:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "q_loss": q_loss.item(),
            "entropy": entropy.item(),
        }

    def train_step(
        self,
        next_state: Tensor,
        next_done: Tensor
    ) -> Optional[Dict[str, float]]:
        """Perform training step with on-policy and off-policy updates.

        Args:
            next_state: (N, 16, 17) state after last transition
            next_done: (N,) done flags for next_state

        Returns:
            Dict with training metrics or None if not enough steps
        """
        if len(self._states) < self.n_steps:
            return None

        # Stack collected trajectory
        states = torch.stack(self._states)      # (T, N, 16, 17)
        actions = torch.stack(self._actions)    # (T, N)
        rewards = torch.stack(self._rewards)    # (T, N)
        dones = torch.stack(self._dones)        # (T, N)
        valid_masks = torch.stack(self._valid_masks)  # (T, N, 4)
        mu = torch.stack(self._mu)              # (T, N)

        # Clear collection
        self._states = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._valid_masks = []
        self._mu = []

        # Store in replay buffer
        self.replay_buffer.push_trajectories(
            states, actions, rewards, dones, valid_masks, mu
        )

        # Compute valid mask for next_state
        from game.moves import compute_valid_mask
        next_valid_mask = compute_valid_mask(next_state, self.device)

        # On-policy update
        metrics = self._update_from_trajectories(
            states, actions, rewards, dones, valid_masks, mu,
            next_state, next_valid_mask, on_policy=True
        )

        # Off-policy updates from replay
        if self.replay_buffer.is_ready(self.buffer_min_size):
            for _ in range(self.replay_ratio):
                (
                    replay_states, replay_actions, replay_rewards,
                    replay_dones, replay_masks, replay_mu
                ) = self.replay_buffer.sample(self.batch_size)

                # For replay, we bootstrap with the last state in trajectory
                # This is a simplification - ideally we'd store next_state too
                replay_next = replay_states[:, -1]  # (B, 16, 17)
                replay_next_mask = replay_masks[:, -1]  # (B, 4)

                # Transpose to (T, B, ...)
                replay_states = replay_states.transpose(0, 1)
                replay_actions = replay_actions.transpose(0, 1)
                replay_rewards = replay_rewards.transpose(0, 1)
                replay_dones = replay_dones.transpose(0, 1)
                replay_masks = replay_masks.transpose(0, 1)
                replay_mu = replay_mu.transpose(0, 1)

                self._update_from_trajectories(
                    replay_states, replay_actions, replay_rewards,
                    replay_dones, replay_masks, replay_mu,
                    replay_next, replay_next_mask, on_policy=False
                )

        self.step_count += 1
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "hidden_layers": self.hidden_layers,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
