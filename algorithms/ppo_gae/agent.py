"""
PPO+GAE (Proximal Policy Optimization with Generalized Advantage Estimation) Agent.

Implements PPO algorithm with:
- Clipped surrogate objective for stable policy updates
- Generalized Advantage Estimation (GAE) with lambda parameter
- Multiple epochs of minibatch updates per rollout

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
"""

from typing import Dict, Any, Optional, List

import torch
import torch.optim as optim
from torch import Tensor

from algorithms.ppo_gae.model import PPOActorCriticNetwork
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class PPOAgent:
    """PPO Agent with GAE for playing 2048.

    PPO Algorithm:
    1. Collect rollout of T steps from parallel environments
    2. Compute advantages using GAE(lambda)
    3. For K epochs:
       - Split data into minibatches
       - Update policy with clipped surrogate objective
       - Update value function (optionally clipped)

    Key equations:
    - GAE: A_t = sum_{l=0}^{inf} (gamma*lambda)^l * delta_{t+l}
           where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
    - Clipped objective: L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
           where ratio = pi_new(a|s) / pi_old(a|s)
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: List[int] = [256, 256],
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        n_steps: int = 128,
        n_epochs: int = 4,
        n_minibatches: int = 4,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        """Initialize PPO agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
                           If None, uses OneHotRepresentation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter (0 = TD(0), 1 = MC)
            clip_ratio: PPO clipping parameter (epsilon)
            n_steps: Number of steps to collect before update
            n_epochs: Number of epochs to train on each batch of data
            n_minibatches: Number of minibatches to split data into
            value_loss_coef: Coefficient for value loss in total loss
            entropy_coef: Coefficient for entropy bonus (exploration)
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        # Get input size from representation
        input_size = self.representation.output_shape()[0]

        # Actor-Critic network
        self.network = PPOActorCriticNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate
        )

        # Rollout storage
        self._states: List[Tensor] = []
        self._actions: List[Tensor] = []
        self._rewards: List[Tensor] = []
        self._dones: List[Tensor] = []
        self._valid_masks: List[Tensor] = []
        self._log_probs: List[Tensor] = []  # Store old log probs for ratio
        self._values: List[Tensor] = []  # Store old values for optional clipping

        # Training state
        self.step_count = 0

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
            training: If True, sample; if False, take argmax (greedy)

        Returns:
            (N,) selected actions
        """
        # Transform state through representation
        with torch.no_grad():
            repr_state = self.representation(state)

        # Get action probabilities
        with torch.no_grad():
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
        else:
            # Greedy action selection
            actions = probs.argmax(dim=1)

        return actions

    def select_action_with_log_prob(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> tuple:
        """Select actions and return log probabilities and values.

        Used during rollout collection to store old policy values.

        Args:
            state: (N, 16, 17) current board states
            valid_mask: (N, 4) valid action masks

        Returns:
            Tuple of (actions, log_probs, values)
        """
        # Transform state through representation
        with torch.no_grad():
            repr_state = self.representation(state)
            log_probs_all, values = self.network.get_action_log_probs_and_value(
                repr_state, valid_mask
            )
            probs = torch.exp(log_probs_all)

            # Sample from the distribution
            probs_safe = probs.clone()
            row_sums = probs_safe.sum(dim=1, keepdim=True)
            no_valid = (row_sums == 0).squeeze(1)
            if no_valid.any():
                probs_safe[no_valid] = valid_mask[no_valid].float()
                probs_safe[no_valid] = probs_safe[no_valid] / probs_safe[no_valid].sum(dim=1, keepdim=True)

            actions = torch.multinomial(probs_safe, 1).squeeze(1)

            # Get log prob of selected action
            action_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        return actions, action_log_probs, values

    def store_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        valid_mask: Tensor,
        log_prob: Tensor,
        value: Tensor
    ) -> None:
        """Store a transition for rollout collection.

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards
            done: (N,) episode termination flags
            valid_mask: (N, 4) valid actions for current states
            log_prob: (N,) log probability of action under old policy
            value: (N,) value estimate of state under old policy
        """
        self._states.append(state.clone())
        self._actions.append(action.clone())
        self._rewards.append(reward.clone())
        self._dones.append(done.clone())
        self._valid_masks.append(valid_mask.clone())
        self._log_probs.append(log_prob.clone())
        self._values.append(value.clone())

    def should_update(self) -> bool:
        """Check if we have enough steps for an update.

        Returns:
            True if we have n_steps collected
        """
        return len(self._states) >= self.n_steps

    def compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: Tensor
    ) -> tuple:
        """Compute Generalized Advantage Estimation.

        GAE(lambda) formula:
        A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)

        Args:
            rewards: (T, N) rewards
            values: (T, N) value estimates
            dones: (T, N) done flags
            next_value: (N,) bootstrap value for last state

        Returns:
            Tuple of (advantages, returns)
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = torch.zeros(N, device=self.device)
        next_val = next_value

        for t in reversed(range(T)):
            # If done, next value is 0
            mask = (~dones[t]).float()
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_val = values[t]

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def train_step(self, next_state: Tensor, next_done: Tensor) -> Optional[Dict[str, float]]:
        """Perform PPO training using collected rollout.

        PPO update:
        1. Compute GAE advantages
        2. For K epochs:
           - Shuffle and split into minibatches
           - For each minibatch:
             - Compute ratio = pi_new / pi_old
             - Compute clipped surrogate objective
             - Update network

        Args:
            next_state: (N, 16, 17) state after the last stored transition
            next_done: (N,) done flags for next_state

        Returns:
            Dict with training metrics or None if not enough steps
        """
        if len(self._states) < self.n_steps:
            return None

        n_games = self._states[0].size(0)
        T = len(self._states)

        # Stack rollout data
        states = torch.stack(self._states)        # (T, N, 16, 17)
        actions = torch.stack(self._actions)      # (T, N)
        rewards = torch.stack(self._rewards)      # (T, N)
        dones = torch.stack(self._dones)          # (T, N)
        valid_masks = torch.stack(self._valid_masks)  # (T, N, 4)
        old_log_probs = torch.stack(self._log_probs)  # (T, N)
        old_values = torch.stack(self._values)    # (T, N)

        # Clear rollout storage
        self._states = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._valid_masks = []
        self._log_probs = []
        self._values = []

        # Normalize rewards for stability
        rewards_flat = rewards.view(-1)
        if rewards_flat.std() > 0:
            rewards_normalized = (rewards - rewards_flat.mean()) / (rewards_flat.std() + 1e-8)
        else:
            rewards_normalized = rewards

        # Get bootstrap value
        with torch.no_grad():
            repr_next = self.representation(next_state)
            next_value = self.network.get_value(repr_next)
            next_value = torch.where(next_done, torch.zeros_like(next_value), next_value)

        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae(
            rewards_normalized, old_values, dones, next_value
        )

        # Flatten for minibatch processing
        # (T, N, ...) -> (T*N, ...)
        flat_states = states.view(-1, 16, 17)
        flat_actions = actions.view(-1)
        flat_valid_masks = valid_masks.view(-1, 4)
        flat_old_log_probs = old_log_probs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)

        # Normalize advantages (important for PPO stability)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # Transform states once
        flat_repr_states = self.representation(flat_states)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0

        batch_size = T * n_games
        minibatch_size = batch_size // self.n_minibatches

        # PPO epochs
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch data
                mb_states = flat_repr_states[mb_indices]
                mb_actions = flat_actions[mb_indices]
                mb_valid_masks = flat_valid_masks[mb_indices]
                mb_old_log_probs = flat_old_log_probs[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]

                # Evaluate actions under current policy
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    mb_states, mb_actions, mb_valid_masks
                )

                # Compute ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (unclipped)
                value_loss = ((values - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy_mean = entropy.mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Check for NaN gradients
                has_nan_grad = False
                for param in self.network.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break

                if not has_nan_grad:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_loss += loss.item()
                n_updates += 1

        self.step_count += 1

        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "hidden_layers": [layer.out_features for layer in self.network.trunk
                             if isinstance(layer, torch.nn.Linear)],
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
