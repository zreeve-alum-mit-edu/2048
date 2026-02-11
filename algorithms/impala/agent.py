"""
IMPALA (Importance Weighted Actor-Learner Architecture) Agent.

Implements IMPALA algorithm with V-trace off-policy correction.
This single-machine implementation simulates the actor-learner
architecture using batched vectorized environments.

Key IMPALA components:
- V-trace for off-policy correction
- Separate actor and learner (simulated via lag)
- Experience queue (simulated via batch collection)

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
Per DEC-0039: Vectorized tensor operations
"""

from typing import Dict, Any, Optional, List

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from algorithms.impala.model import IMPALANetwork
from algorithms.impala.vtrace import compute_vtrace
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class IMPALAAgent:
    """IMPALA Agent for playing 2048.

    This implementation simulates IMPALA's actor-learner architecture
    on a single machine. The key insight is that V-trace allows us to
    learn from experience collected by a slightly stale policy (the
    "behavior policy") while updating the current "target policy".

    In this simulation:
    - Actors collect experience with the current network (behavior policy)
    - Learner updates with V-trace using stored behavior logits
    - Policy lag is simulated by storing behavior logits with trajectories
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: List[int] = [256, 256],
        learning_rate: float = 0.0006,
        gamma: float = 0.99,
        n_steps: int = 20,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 40.0,
    ):
        """Initialize IMPALA agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            n_steps: Number of steps for trajectories
            rho_bar: V-trace rho truncation threshold
            c_bar: V-trace c truncation threshold
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.device = device
        self.gamma = gamma
        self.n_steps = n_steps
        self.rho_bar = rho_bar
        self.c_bar = c_bar
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
        self.network = IMPALANetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Optimizer (RMSprop is traditional for IMPALA)
        self.optimizer = optim.RMSprop(
            self.network.parameters(),
            lr=learning_rate,
            eps=0.01,
            alpha=0.99,
        )

        # Rollout storage
        self._states: List[Tensor] = []
        self._actions: List[Tensor] = []
        self._rewards: List[Tensor] = []
        self._dones: List[Tensor] = []
        self._valid_masks: List[Tensor] = []
        self._behavior_logits: List[Tensor] = []  # Store behavior policy logits

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
            training: If True, sample; if False, greedy

        Returns:
            (N,) selected actions
        """
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
        valid_mask: Tensor,
        behavior_logits: Tensor
    ) -> None:
        """Store a transition with behavior policy logits.

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards
            done: (N,) episode termination flags
            valid_mask: (N, 4) valid actions for current states
            behavior_logits: (N, 4) behavior policy logits at action selection
        """
        self._states.append(state.clone())
        self._actions.append(action.clone())
        self._rewards.append(reward.clone())
        self._dones.append(done.clone())
        self._valid_masks.append(valid_mask.clone())
        self._behavior_logits.append(behavior_logits.clone())

    def get_behavior_logits(self, state: Tensor, valid_mask: Tensor) -> Tensor:
        """Get behavior policy logits for storing.

        Args:
            state: (N, 16, 17) current states
            valid_mask: (N, 4) valid action masks

        Returns:
            (N, 4) masked logits from current policy (behavior)
        """
        with torch.no_grad():
            repr_state = self.representation(state)
            logits, _ = self.network.get_policy_logits_and_value(repr_state, valid_mask)
        return logits

    def train_step(
        self,
        next_state: Tensor,
        next_done: Tensor
    ) -> Optional[Dict[str, float]]:
        """Perform one training step using V-trace.

        Args:
            next_state: (N, 16, 17) state after last transition
            next_done: (N,) done flags for next_state

        Returns:
            Dict with training metrics or None if not enough steps
        """
        if len(self._states) < self.n_steps:
            return None

        n_games = self._states[0].size(0)
        n_steps = len(self._states)

        # Stack collected trajectory
        states = torch.stack(self._states)           # (T, N, 16, 17)
        actions = torch.stack(self._actions)         # (T, N)
        rewards = torch.stack(self._rewards)         # (T, N)
        dones = torch.stack(self._dones)             # (T, N)
        valid_masks = torch.stack(self._valid_masks) # (T, N, 4)
        behavior_logits = torch.stack(self._behavior_logits)  # (T, N, 4)

        # Clear collection
        self._states = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._valid_masks = []
        self._behavior_logits = []

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
        repr_states = repr_flat.view(n_steps, n_games, repr_size)

        # Get current policy logits and values
        flat_repr = repr_states.view(-1, repr_size)
        flat_masks = valid_masks.view(-1, 4)

        target_logits, values = self.network.get_policy_logits_and_value(flat_repr, flat_masks)
        target_logits = target_logits.view(n_steps, n_games, 4)
        values = values.view(n_steps, n_games)

        # Get bootstrap value
        from game.moves import compute_valid_mask
        next_valid_mask = compute_valid_mask(next_state, self.device)

        repr_next = self.representation(next_state)
        _, bootstrap_value = self.network.get_policy_logits_and_value(repr_next, next_valid_mask)
        bootstrap_value = torch.where(next_done, torch.zeros_like(bootstrap_value), bootstrap_value)

        # Compute log probabilities for actions taken
        # Target policy log probs
        target_log_probs_all = F.log_softmax(target_logits, dim=-1)
        target_log_probs = target_log_probs_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # Behavior policy log probs
        behavior_log_probs_all = F.log_softmax(behavior_logits, dim=-1)
        behavior_log_probs = behavior_log_probs_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # Compute V-trace targets and advantages
        with torch.no_grad():
            vs_targets, advantages = compute_vtrace(
                rewards_norm, dones, values.detach(), bootstrap_value.detach(),
                target_log_probs.detach(), behavior_log_probs,
                gamma=self.gamma, rho_bar=self.rho_bar, c_bar=self.c_bar
            )

        # Value loss: (V(s) - v_s)^2
        value_loss = ((values - vs_targets) ** 2).mean()

        # Policy loss: -advantages * log_pi(a|s)
        policy_loss = -(advantages * target_log_probs).mean()

        # Entropy bonus
        target_probs = F.softmax(target_logits, dim=-1)
        entropy_per_state = -(target_probs * target_log_probs_all).sum(dim=-1)
        valid_entropy = entropy_per_state[entropy_per_state.isfinite()]
        entropy = valid_entropy.mean() if valid_entropy.numel() > 0 else torch.tensor(0.0, device=self.device)

        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients
        has_nan = any(
            param.grad is not None and torch.isnan(param.grad).any()
            for param in self.network.parameters()
        )

        if not has_nan:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

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
