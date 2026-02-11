"""
A3C (Asynchronous Advantage Actor-Critic) Agent.

Implements A3C algorithm with simulated async workers via vectorized
environments. Each "worker" is a slice of the batch, with gradient
accumulation across workers before applying updates.

Key A3C concepts implemented:
- Multiple workers collecting experience in parallel
- Gradient accumulation from all workers
- Shared global network (simulated via single network)
- n-step returns for advantage estimation

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
Per DEC-0039: Vectorized tensor operations
"""

from typing import Dict, Any, Optional, List

import torch
import torch.optim as optim
from torch import Tensor

from algorithms.a3c.model import A3CNetwork
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class A3CAgent:
    """A3C Agent for playing 2048.

    This implementation simulates A3C's async behavior using vectorized
    environments. The batch is divided into "worker groups", each
    collecting trajectories that are combined for gradient computation.

    Key differences from A2C:
    - Simulated async workers (via batch slicing)
    - Gradient accumulation across workers before update
    - Worker-specific entropy bonus scaling
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: List[int] = [256, 256],
        learning_rate: float = 0.0007,
        gamma: float = 0.99,
        n_steps: int = 5,
        num_workers: int = 4,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 40.0,
    ):
        """Initialize A3C agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
                           If None, uses OneHotRepresentation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            n_steps: Number of steps for n-step returns
            num_workers: Number of simulated async workers
            value_loss_coef: Coefficient for value loss in total loss
            entropy_coef: Coefficient for entropy bonus (exploration)
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.device = device
        self.gamma = gamma
        self.n_steps = n_steps
        self.num_workers = num_workers
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

        # Shared global network (workers share this)
        self.network = A3CNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Shared optimizer (RMSprop is traditional for A3C, but Adam works well)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

        # Rollout storage for n-step collection (per worker)
        self._states: List[Tensor] = []
        self._actions: List[Tensor] = []
        self._rewards: List[Tensor] = []
        self._dones: List[Tensor] = []
        self._valid_masks: List[Tensor] = []

        # Training state
        self.step_count = 0
        self.hidden_layers = hidden_layers

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

    def train_step(self, next_state: Tensor, next_done: Tensor) -> Optional[Dict[str, float]]:
        """Perform one training step using collected n-step rollout.

        A3C update with simulated async workers:
        - Divide batch into worker groups
        - Compute gradients per worker
        - Accumulate gradients
        - Apply single update

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
        flat_states = states.view(-1, 16, 17)
        repr_flat_states = self.representation(flat_states)
        repr_size = repr_flat_states.size(-1)
        repr_states = repr_flat_states.view(n_steps, n_games, repr_size)

        # Get bootstrap value for n-step return
        with torch.no_grad():
            repr_next = self.representation(next_state)
            next_value = self.network.get_value(repr_next)
            next_value = torch.where(next_done, torch.zeros_like(next_value), next_value)

        # Normalize rewards for stability
        rewards_flat = rewards.view(-1)
        if rewards_flat.std() > 0:
            rewards_normalized = (rewards - rewards_flat.mean()) / (rewards_flat.std() + 1e-8)
        else:
            rewards_normalized = rewards

        # Compute n-step returns (backwards)
        returns = torch.zeros(n_steps, n_games, device=self.device)
        R = next_value

        for t in reversed(range(n_steps)):
            R = torch.where(dones[t], torch.zeros_like(R), R)
            R = rewards_normalized[t] + self.gamma * R
            returns[t] = R

        # Flatten for network forward pass
        flat_repr_states = repr_states.view(-1, repr_size)
        flat_valid_masks = valid_masks.view(-1, 4)
        flat_actions = actions.view(-1)
        flat_returns = returns.view(-1)

        # A3C: Simulate async workers by processing in groups
        # Each worker computes its own loss, then we average
        worker_size = n_games // self.num_workers
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        # Zero gradients once before accumulation
        self.optimizer.zero_grad()

        for worker_idx in range(self.num_workers):
            # Get worker's slice of the batch
            start_idx = worker_idx * worker_size * n_steps
            end_idx = (worker_idx + 1) * worker_size * n_steps

            worker_states = flat_repr_states[start_idx:end_idx]
            worker_masks = flat_valid_masks[start_idx:end_idx]
            worker_actions = flat_actions[start_idx:end_idx]
            worker_returns = flat_returns[start_idx:end_idx]

            # Forward pass for this worker
            log_probs, values = self.network.get_action_log_probs_and_value(
                worker_states, worker_masks
            )

            # Get log prob of taken action
            action_log_probs = log_probs.gather(1, worker_actions.unsqueeze(1)).squeeze(1)

            # Compute advantages
            advantages = worker_returns - values.detach()

            # Policy loss
            policy_loss = -(advantages * action_log_probs).mean()

            # Value loss
            value_loss = ((values - worker_returns) ** 2).mean()

            # Entropy bonus
            probs = torch.exp(log_probs)
            entropy_per_state = -(probs * log_probs).sum(dim=1)
            valid_entropy = entropy_per_state[entropy_per_state.isfinite()]
            entropy = valid_entropy.mean() if valid_entropy.numel() > 0 else torch.tensor(0.0, device=self.device)

            # Worker loss
            worker_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            # Accumulate gradients (scaled by 1/num_workers for averaging)
            (worker_loss / self.num_workers).backward()

            total_loss += worker_loss.item() / self.num_workers
            total_policy_loss += policy_loss.item() / self.num_workers
            total_value_loss += value_loss.item() / self.num_workers
            total_entropy += entropy.item() / self.num_workers

        # Check for NaN gradients
        has_nan_grad = False
        for param in self.network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break

        if not has_nan_grad:
            # Gradient clipping (A3C typically uses larger max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
        else:
            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        self.step_count += 1

        return {
            "loss": total_loss,
            "policy_loss": total_policy_loss,
            "value_loss": total_value_loss,
            "entropy": total_entropy,
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
            "hidden_layers": self.hidden_layers,
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
