"""
PPO with Value Function Clipping Agent.

Extends PPO+GAE with value function clipping to prevent large value updates.
This can help stabilize training when the value function is prone to
overshooting.

Key difference from PPO+GAE:
- Value loss is clipped: V_new is clipped to [V_old - eps, V_old + eps]
- Final value loss is max of unclipped and clipped loss

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


class PPOValueClipAgent:
    """PPO Agent with Value Clipping for playing 2048.

    Same as PPO+GAE but with clipped value loss:
    - V_clipped = V_old + clip(V_new - V_old, -eps, eps)
    - Loss = max((V_new - R)^2, (V_clipped - R)^2)

    This prevents the value function from changing too much in a single update.
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
        value_clip_ratio: float = 0.2,  # Can be different from policy clip
        n_steps: int = 128,
        n_epochs: int = 4,
        n_minibatches: int = 4,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        """Initialize PPO+Value Clip agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO policy clipping parameter
            value_clip_ratio: Value function clipping parameter
            n_steps: Number of steps to collect before update
            n_epochs: Number of epochs to train on each batch
            n_minibatches: Number of minibatches per epoch
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_clip_ratio = value_clip_ratio
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

        # Actor-Critic network (reuse PPO+GAE architecture)
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
        self._log_probs: List[Tensor] = []
        self._values: List[Tensor] = []

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
            training: If True, sample; if False, greedy

        Returns:
            (N,) selected actions
        """
        with torch.no_grad():
            repr_state = self.representation(state)
            probs = self.network.get_action_probs(repr_state, valid_mask)

        if training:
            probs_safe = probs.clone()
            row_sums = probs_safe.sum(dim=1, keepdim=True)
            no_valid = (row_sums == 0).squeeze(1)
            if no_valid.any():
                probs_safe[no_valid] = valid_mask[no_valid].float()
                probs_safe[no_valid] = probs_safe[no_valid] / probs_safe[no_valid].sum(dim=1, keepdim=True)
            actions = torch.multinomial(probs_safe, 1).squeeze(1)
        else:
            actions = probs.argmax(dim=1)

        return actions

    def select_action_with_log_prob(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> tuple:
        """Select actions and return log probabilities and values.

        Args:
            state: (N, 16, 17) current board states
            valid_mask: (N, 4) valid action masks

        Returns:
            Tuple of (actions, log_probs, values)
        """
        with torch.no_grad():
            repr_state = self.representation(state)
            log_probs_all, values = self.network.get_action_log_probs_and_value(
                repr_state, valid_mask
            )
            probs = torch.exp(log_probs_all)

            probs_safe = probs.clone()
            row_sums = probs_safe.sum(dim=1, keepdim=True)
            no_valid = (row_sums == 0).squeeze(1)
            if no_valid.any():
                probs_safe[no_valid] = valid_mask[no_valid].float()
                probs_safe[no_valid] = probs_safe[no_valid] / probs_safe[no_valid].sum(dim=1, keepdim=True)

            actions = torch.multinomial(probs_safe, 1).squeeze(1)
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
        """Store a transition for rollout collection."""
        self._states.append(state.clone())
        self._actions.append(action.clone())
        self._rewards.append(reward.clone())
        self._dones.append(done.clone())
        self._valid_masks.append(valid_mask.clone())
        self._log_probs.append(log_prob.clone())
        self._values.append(value.clone())

    def should_update(self) -> bool:
        """Check if we have enough steps for an update."""
        return len(self._states) >= self.n_steps

    def compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: Tensor
    ) -> tuple:
        """Compute Generalized Advantage Estimation."""
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)

        gae = torch.zeros(N, device=self.device)
        next_val = next_value

        for t in reversed(range(T)):
            mask = (~dones[t]).float()
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_val = values[t]

        returns = advantages + values
        return advantages, returns

    def train_step(self, next_state: Tensor, next_done: Tensor) -> Optional[Dict[str, float]]:
        """Perform PPO+Value Clip training.

        Key difference: Value loss uses clipping.

        Args:
            next_state: (N, 16, 17) state after last transition
            next_done: (N,) done flags for next_state

        Returns:
            Dict with training metrics or None if not enough steps
        """
        if len(self._states) < self.n_steps:
            return None

        n_games = self._states[0].size(0)
        T = len(self._states)

        # Stack rollout data
        states = torch.stack(self._states)
        actions = torch.stack(self._actions)
        rewards = torch.stack(self._rewards)
        dones = torch.stack(self._dones)
        valid_masks = torch.stack(self._valid_masks)
        old_log_probs = torch.stack(self._log_probs)
        old_values = torch.stack(self._values)

        # Clear storage
        self._states = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._valid_masks = []
        self._log_probs = []
        self._values = []

        # Normalize rewards
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

        # Compute GAE
        advantages, returns = self.compute_gae(
            rewards_normalized, old_values, dones, next_value
        )

        # Flatten
        flat_states = states.view(-1, 16, 17)
        flat_actions = actions.view(-1)
        flat_valid_masks = valid_masks.view(-1, 4)
        flat_old_log_probs = old_log_probs.view(-1)
        flat_old_values = old_values.view(-1)  # Need old values for clipping
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # Transform states
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
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_states = flat_repr_states[mb_indices]
                mb_actions = flat_actions[mb_indices]
                mb_valid_masks = flat_valid_masks[mb_indices]
                mb_old_log_probs = flat_old_log_probs[mb_indices]
                mb_old_values = flat_old_values[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]

                # Evaluate actions
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    mb_states, mb_actions, mb_valid_masks
                )

                # Policy loss (same as PPO+GAE)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # VALUE LOSS WITH CLIPPING (key difference)
                # Unclipped value loss
                value_loss_unclipped = (values - mb_returns) ** 2

                # Clipped value: V_clipped = V_old + clip(V_new - V_old, -eps, eps)
                values_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values,
                    -self.value_clip_ratio,
                    self.value_clip_ratio
                )
                value_loss_clipped = (values_clipped - mb_returns) ** 2

                # Take max of clipped and unclipped (pessimistic)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy
                entropy_mean = entropy.mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                has_nan_grad = False
                for param in self.network.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break

                if not has_nan_grad:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

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
        """Save agent checkpoint."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "hidden_layers": [layer.out_features for layer in self.network.trunk
                             if isinstance(layer, torch.nn.Linear)],
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
