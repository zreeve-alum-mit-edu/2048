"""
A2C (Advantage Actor-Critic) Agent.

Implements synchronous Advantage Actor-Critic algorithm:
- Actor: learns policy pi(a|s)
- Critic: learns value function V(s)
- Advantage: A(s,a) = Q(s,a) - V(s) = r + gamma*V(s') - V(s)

Uses n-step returns for advantage estimation and entropy bonus for exploration.

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
"""

from typing import Dict, Any, Optional, List

import torch
import torch.optim as optim
from torch import Tensor

from algorithms.a2c.model import ActorCriticNetwork
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class A2CAgent:
    """A2C Agent for playing 2048.

    Algorithm:
    1. Collect n-step trajectories from parallel environments
    2. Compute n-step returns and advantages
    3. Update policy using advantage-weighted log probabilities
    4. Update value function using TD error

    Loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: list = [256, 256],
        learning_rate: float = 0.0007,
        gamma: float = 0.99,
        n_steps: int = 5,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """Initialize A2C agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
                           If None, uses OneHotRepresentation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            n_steps: Number of steps for n-step returns
            value_loss_coef: Coefficient for value loss in total loss
            entropy_coef: Coefficient for entropy bonus (exploration)
        """
        self.device = device
        self.gamma = gamma
        self.n_steps = n_steps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        # Get input size from representation
        input_size = self.representation.output_shape()[0]

        # Actor-Critic network
        self.network = ActorCriticNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate
        )

        # Rollout storage for n-step collection
        self._states: List[Tensor] = []
        self._actions: List[Tensor] = []
        self._rewards: List[Tensor] = []
        self._dones: List[Tensor] = []
        self._valid_masks: List[Tensor] = []

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

    def store_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        valid_mask: Tensor
    ) -> None:
        """Store a transition for n-step collection.

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards
            done: (N,) episode termination flags
            valid_mask: (N, 4) valid actions for current states
        """
        self._states.append(state.clone())
        self._actions.append(action.clone())
        self._rewards.append(reward.clone())
        self._dones.append(done.clone())
        self._valid_masks.append(valid_mask.clone())

    def should_update(self) -> bool:
        """Check if we have enough steps for an update.

        Returns:
            True if we have n_steps collected
        """
        return len(self._states) >= self.n_steps

    def train_step(self, next_state: Tensor, next_done: Tensor) -> Optional[Dict[str, float]]:
        """Perform one training step using collected n-step rollout.

        A2C update:
        - Compute n-step returns: R_t = sum_{i=0}^{n-1} gamma^i * r_{t+i} + gamma^n * V(s_{t+n})
        - Advantage: A_t = R_t - V(s_t)
        - Policy loss: -log(pi(a|s)) * A
        - Value loss: (V(s) - R)^2
        - Entropy bonus: -sum(pi * log(pi))

        Args:
            next_state: (N, 16, 17) state after the last stored transition
            next_done: (N,) done flags for next_state

        Returns:
            Dict with training metrics or None if not enough steps
        """
        if len(self._states) < self.n_steps:
            return None

        n_games = self._states[0].size(0)
        n_steps = len(self._states)

        # Stack rollout data
        states = torch.stack(self._states)      # (T, N, 16, 17)
        actions = torch.stack(self._actions)    # (T, N)
        rewards = torch.stack(self._rewards)    # (T, N)
        dones = torch.stack(self._dones)        # (T, N)
        valid_masks = torch.stack(self._valid_masks)  # (T, N, 4)

        # Clear rollout storage
        self._states = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._valid_masks = []

        # Transform states through representation
        # Reshape for batch processing: (T, N, 16, 17) -> (T*N, 16, 17)
        flat_states = states.view(-1, 16, 17)
        repr_flat_states = self.representation(flat_states)
        # Reshape back: (T*N, repr_size) -> (T, N, repr_size)
        repr_size = repr_flat_states.size(-1)
        repr_states = repr_flat_states.view(n_steps, n_games, repr_size)

        # Get bootstrap value for n-step return
        with torch.no_grad():
            repr_next = self.representation(next_state)
            next_value = self.network.get_value(repr_next)
            # Zero out value for terminal states
            next_value = torch.where(next_done, torch.zeros_like(next_value), next_value)

        # Normalize rewards for stability (2048 rewards can be very large)
        # Use running mean/std or simple normalization
        rewards_flat = rewards.view(-1)
        if rewards_flat.std() > 0:
            rewards_normalized = (rewards - rewards_flat.mean()) / (rewards_flat.std() + 1e-8)
        else:
            rewards_normalized = rewards

        # Compute n-step returns (backwards) using normalized rewards
        returns = torch.zeros(n_steps, n_games, device=self.device)
        R = next_value

        for t in reversed(range(n_steps)):
            # If done, R resets to 0 (terminal)
            R = torch.where(dones[t], torch.zeros_like(R), R)
            R = rewards_normalized[t] + self.gamma * R
            returns[t] = R

        # Flatten for network forward pass
        flat_repr_states = repr_states.view(-1, repr_size)  # (T*N, repr_size)
        flat_valid_masks = valid_masks.view(-1, 4)           # (T*N, 4)
        flat_actions = actions.view(-1)                      # (T*N,)
        flat_returns = returns.view(-1)                      # (T*N,)

        # Forward pass
        log_probs, values = self.network.get_action_log_probs_and_value(
            flat_repr_states, flat_valid_masks
        )

        # Get log prob of taken action
        action_log_probs = log_probs.gather(1, flat_actions.unsqueeze(1)).squeeze(1)

        # Compute advantages
        advantages = flat_returns - values.detach()

        # Policy loss (negative because we want to maximize)
        policy_loss = -(advantages * action_log_probs).mean()

        # Value loss (MSE)
        value_loss = ((values - flat_returns) ** 2).mean()

        # Entropy bonus (for exploration)
        # entropy = -sum(p * log(p))
        probs = torch.exp(log_probs)
        # Only compute entropy for valid actions
        entropy_per_state = -(probs * log_probs).sum(dim=1)
        # Mask out states where we have numerical issues (all -inf log probs)
        valid_entropy = entropy_per_state[entropy_per_state.isfinite()]
        entropy = valid_entropy.mean() if valid_entropy.numel() > 0 else torch.tensor(0.0, device=self.device)

        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients and skip update if found
        has_nan_grad = False
        for param in self.network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break

        if not has_nan_grad:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()
        else:
            # Zero out NaN gradients to prevent corruption
            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "avg_value": values.mean().item(),
            "avg_return": flat_returns.mean().item(),
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
            # Save architecture info for loading
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
