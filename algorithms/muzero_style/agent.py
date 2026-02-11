"""
MuZero-style Agent.

Implements MuZero-style training with learned dynamics model.

Training:
1. Play games with MCTS using learned model
2. Store trajectories in buffer
3. Train all networks jointly on sampled trajectories

This is a simplified version without:
- Reanalysis (updating stored trajectories with newer model)
- Search parallelism

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
"""

from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from algorithms.muzero_style.model import MuZeroNetworks
from algorithms.muzero_style.buffer import TrajectoryBuffer, Trajectory
from algorithms.shared.mcts_base import MCTSBase, MCTSConfig, DynamicsProvider
from game.env import GameEnv
from game.moves import compute_valid_mask
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class LearnedDynamicsProvider(DynamicsProvider):
    """Dynamics provider that uses learned dynamics model.

    For MuZero, we use the learned representation, dynamics, and
    prediction networks instead of the real environment.
    """

    def __init__(
        self,
        networks: MuZeroNetworks,
        representation: Representation,
        device: torch.device,
    ):
        """Initialize dynamics provider.

        Args:
            networks: MuZero networks
            representation: State representation module
            device: PyTorch device
        """
        self.networks = networks
        self.representation = representation
        self.device = device

        # For tracking valid masks in learned dynamics
        # MuZero doesn't naturally know valid moves, so we use a heuristic
        self._last_valid_mask: Optional[Tensor] = None

    def set_valid_mask(self, valid_mask: Tensor) -> None:
        """Set the valid mask for the current state.

        Since learned dynamics don't know valid moves, we need
        to track this externally during MCTS.

        Args:
            valid_mask: (4,) boolean mask
        """
        self._last_valid_mask = valid_mask.clone()

    def get_initial_state(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        """Get initial hidden state from observation."""
        with torch.no_grad():
            obs_batch = observation.unsqueeze(0)
            repr_obs = self.representation(obs_batch)

            # Get hidden state
            hidden_state = self.networks.representation(repr_obs)
            hidden_state = hidden_state.squeeze(0)

            # Get valid mask from real observation
            valid_mask = compute_valid_mask(obs_batch, self.device).squeeze(0)

        self._last_valid_mask = valid_mask.clone()
        return hidden_state, valid_mask

    def step(
        self,
        hidden_state: Tensor,
        action: int
    ) -> Tuple[Tensor, float, bool, Tensor]:
        """Take a step using learned dynamics."""
        with torch.no_grad():
            # Add batch dimension
            hidden_batch = hidden_state.unsqueeze(0)
            action_tensor = torch.tensor([action], device=self.device)

            # Use last valid mask (MuZero doesn't predict valid moves)
            valid_mask = self._last_valid_mask
            if valid_mask is None:
                valid_mask = torch.ones(4, dtype=torch.bool, device=self.device)

            # Dynamics step
            next_hidden, reward, policy, value = self.networks.recurrent_inference(
                hidden_batch, action_tensor, valid_mask.unsqueeze(0)
            )

            next_hidden = next_hidden.squeeze(0)
            reward = reward.squeeze(0).item()

            # Heuristic for done: if value is very low, might be terminal
            # In practice, we don't know done from learned dynamics
            done = False

            # Valid mask stays the same (heuristic - we don't predict it)
            # In full MuZero, you might predict this or use special handling

        return next_hidden, reward, done, valid_mask

    def get_policy_value(
        self,
        hidden_state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, float]:
        """Get policy and value from prediction network."""
        with torch.no_grad():
            hidden_batch = hidden_state.unsqueeze(0)
            valid_batch = valid_mask.unsqueeze(0)

            # Get policy and value
            policy_logits, value = self.networks.prediction(hidden_batch)

            # Mask and normalize policy
            masked_logits = policy_logits.clone()
            masked_logits[~valid_batch] = float('-inf')
            policy = F.softmax(masked_logits, dim=1).squeeze(0)

            return policy, value.squeeze(0).item()


class MuZeroAgent:
    """MuZero-style Agent for playing 2048.

    Uses learned dynamics for MCTS planning.
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_size: int = 256,
        representation_layers: list = [256],
        dynamics_layers: list = [256],
        prediction_layers: list = [256],
        learning_rate: float = 0.001,
        gamma: float = 0.997,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25,
        temperature: float = 1.0,
        temperature_drop_step: int = 50000,
        buffer_capacity: int = 10000,
        td_steps: int = 10,
        unroll_steps: int = 5,
        batch_size: int = 64,
        value_loss_weight: float = 0.25,
        policy_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
    ):
        """Initialize MuZero agent."""
        self.device = device
        self.gamma = gamma
        self.unroll_steps = unroll_steps
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.temperature_drop_step = temperature_drop_step

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        input_size = self.representation.output_shape()[0]

        # MuZero networks
        self.networks = MuZeroNetworks(
            input_size=input_size,
            hidden_size=hidden_size,
            representation_layers=representation_layers,
            dynamics_layers=dynamics_layers,
            prediction_layers=prediction_layers,
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.networks.parameters(), lr=learning_rate
        )

        # MCTS configuration
        self.mcts_config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            exploration_fraction=exploration_fraction,
            temperature=temperature,
            temperature_drop_step=temperature_drop_step,
        )

        # Dynamics provider for MCTS
        self.dynamics = LearnedDynamicsProvider(
            self.networks, self.representation, device
        )

        # MCTS
        self.mcts = MCTSBase(self.dynamics, self.mcts_config, device)

        # Trajectory buffer
        self.buffer = TrajectoryBuffer(
            capacity=buffer_capacity,
            device=device,
            gamma=gamma,
            td_steps=td_steps,
        )

        # Training state
        self.step_count = 0
        self.hidden_size = hidden_size

    def select_action(
        self,
        observation: Tensor,
        valid_mask: Tensor,
        training: bool = True
    ) -> Tuple[int, Tensor, float]:
        """Select action using MCTS with learned dynamics.

        Args:
            observation: (16, 17) game state
            valid_mask: (4,) valid action mask
            training: Whether to use exploration

        Returns:
            Tuple of (action, policy_target, value_estimate)
        """
        # Set valid mask for dynamics provider
        self.dynamics.set_valid_mask(valid_mask)

        # Determine temperature
        if training:
            if self.step_count >= self.mcts_config.temperature_drop_step:
                temperature = 0.5
            else:
                temperature = self.mcts_config.temperature
        else:
            temperature = 0.0

        # Run MCTS
        root = self.mcts.search(
            observation,
            add_exploration_noise=training
        )

        # Get action and policy target
        action = self.mcts.select_action(root, temperature)
        policy_target = self.mcts.get_policy_target(root)
        value_estimate = root.value

        return action, policy_target, value_estimate

    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Store a complete trajectory in the buffer."""
        self.buffer.push(trajectory)

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step.

        Trains all networks jointly using sampled trajectories.

        Returns:
            Dict with training metrics or None if buffer not ready
        """
        if not self.buffer.is_ready(10):  # Need some trajectories
            return None

        # Sample batch
        (observations, actions, target_values, target_rewards,
         target_policies, valid_masks) = self.buffer.sample(
            self.batch_size, self.unroll_steps
        )

        # Initial inference
        repr_obs = self.representation(observations)
        hidden_states, policies, values = self.networks.initial_inference(
            repr_obs, valid_masks[:, 0]
        )

        # Compute losses
        policy_loss = self._cross_entropy_loss(policies, target_policies[:, 0])
        value_loss = F.mse_loss(values, target_values[:, 0])
        reward_loss = torch.tensor(0.0, device=self.device)

        # Unroll dynamics
        for k in range(self.unroll_steps):
            hidden_states, pred_rewards, policies, values = \
                self.networks.recurrent_inference(
                    hidden_states,
                    actions[:, k],
                    valid_masks[:, k + 1]
                )

            # Accumulate losses
            policy_loss += self._cross_entropy_loss(
                policies, target_policies[:, k + 1]
            )
            value_loss += F.mse_loss(values, target_values[:, k + 1])
            reward_loss += F.mse_loss(pred_rewards, target_rewards[:, k])

        # Average losses over unroll steps
        policy_loss = policy_loss / (self.unroll_steps + 1)
        value_loss = value_loss / (self.unroll_steps + 1)
        reward_loss = reward_loss / self.unroll_steps if self.unroll_steps > 0 else reward_loss

        # Total loss
        loss = (self.policy_loss_weight * policy_loss +
                self.value_loss_weight * value_loss +
                self.reward_loss_weight * reward_loss)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.networks.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "reward_loss": reward_loss.item(),
        }

    def _cross_entropy_loss(
        self,
        predicted: Tensor,
        target: Tensor
    ) -> Tensor:
        """Compute cross-entropy loss for policies.

        Args:
            predicted: (N, 4) predicted probabilities
            target: (N, 4) target probabilities

        Returns:
            Scalar loss
        """
        # -sum(target * log(predicted + eps))
        log_pred = torch.log(predicted + 1e-8)
        return -(target * log_pred).sum(dim=1).mean()

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "networks_state_dict": self.networks.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "hidden_size": self.hidden_size,
            "mcts_config": {
                "num_simulations": self.mcts_config.num_simulations,
                "c_puct": self.mcts_config.c_puct,
                "dirichlet_alpha": self.mcts_config.dirichlet_alpha,
                "exploration_fraction": self.mcts_config.exploration_fraction,
                "temperature": self.mcts_config.temperature,
            }
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.networks.load_state_dict(checkpoint["networks_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
