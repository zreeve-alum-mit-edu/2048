"""
MCTS+Learned Agent.

Implements MCTS with learned policy and value networks using
the real game environment for dynamics.

Training:
- Self-play games with MCTS action selection
- Policy targets from MCTS visit counts
- Value targets from actual game returns

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
"""

from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from algorithms.mcts_learned.model import PolicyValueNetwork
from algorithms.shared.mcts_base import MCTSBase, MCTSConfig, DynamicsProvider
from game.env import GameEnv
from game.moves import compute_valid_mask
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class RealEnvDynamicsProvider(DynamicsProvider):
    """Dynamics provider that uses the real game environment.

    For MCTS+learned, we use the actual game mechanics for state transitions
    but learned networks for policy and value estimation.
    """

    def __init__(
        self,
        network: PolicyValueNetwork,
        representation: Representation,
        device: torch.device,
    ):
        """Initialize dynamics provider.

        Args:
            network: Policy-value network
            representation: State representation module
            device: PyTorch device
        """
        self.network = network
        self.representation = representation
        self.device = device

        # Create single-game environment for tree search
        self._env: Optional[GameEnv] = None
        self._current_state: Optional[Tensor] = None

    def set_state(self, state: Tensor) -> None:
        """Set the current state for dynamics simulation.

        Args:
            state: (16, 17) game state
        """
        self._current_state = state.clone()

        # Create environment if needed
        if self._env is None:
            self._env = GameEnv(n_games=1, device=self.device)

        # Set environment state
        self._env._state = state.unsqueeze(0).clone()

    def get_initial_state(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        """Get initial state and valid actions from observation."""
        self._current_state = observation.clone()

        valid_mask = compute_valid_mask(
            observation.unsqueeze(0), self.device
        ).squeeze(0)

        return observation, valid_mask

    def step(
        self,
        state: Tensor,
        action: int
    ) -> Tuple[Tensor, float, bool, Tensor]:
        """Take a step using the real environment.

        Creates a temporary environment to simulate the step.
        """
        # Create temp env with current state
        temp_env = GameEnv(n_games=1, device=self.device)
        temp_env._state = state.unsqueeze(0).clone()

        # Take action
        action_tensor = torch.tensor([action], device=self.device)
        result = temp_env.step(action_tensor)

        next_state = result.next_state.squeeze(0)
        reward = result.merge_reward.squeeze(0).item()
        done = result.done.squeeze(0).item()
        valid_mask = result.valid_mask.squeeze(0)

        return next_state, reward, done, valid_mask

    def get_policy_value(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, float]:
        """Get policy and value from neural network."""
        with torch.no_grad():
            # Add batch dimension
            state_batch = state.unsqueeze(0)
            valid_batch = valid_mask.unsqueeze(0)

            # Get representation
            repr_state = self.representation(state_batch)

            # Get policy and value
            policy, value = self.network.get_policy_and_value(
                repr_state, valid_batch
            )

            return policy.squeeze(0), value.squeeze(0).item()


class MCTSLearnedAgent:
    """MCTS+Learned Agent for playing 2048.

    Uses MCTS with:
    - Real environment dynamics for state transitions
    - Learned policy network for action priors
    - Learned value network for leaf evaluation
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: list = [256, 256],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25,
        temperature: float = 1.0,
        temperature_drop_step: int = 50000,
        buffer_capacity: int = 100000,
        buffer_min_size: int = 1000,
        batch_size: int = 64,
        value_loss_weight: float = 1.0,
        policy_loss_weight: float = 1.0,
    ):
        """Initialize MCTS+Learned agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for value targets
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for PUCT formula
            dirichlet_alpha: Dirichlet noise parameter
            exploration_fraction: Fraction of Dirichlet noise to add
            temperature: Temperature for action selection
            temperature_drop_step: Step at which to reduce temperature
            buffer_capacity: Replay buffer capacity
            buffer_min_size: Minimum buffer size before training
            batch_size: Batch size for training
            value_loss_weight: Weight for value loss
            policy_loss_weight: Weight for policy loss
        """
        self.device = device
        self.gamma = gamma
        self.buffer_capacity = buffer_capacity
        self.buffer_min_size = buffer_min_size
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.temperature_drop_step = temperature_drop_step

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        input_size = self.representation.output_shape()[0]

        # Policy-value network
        self.network = PolicyValueNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers,
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate
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

        # Create dynamics provider
        self.dynamics = RealEnvDynamicsProvider(
            self.network, self.representation, device
        )

        # Create MCTS
        self.mcts = MCTSBase(self.dynamics, self.mcts_config, device)

        # Replay buffer for training data
        # Stores (state, policy_target, value_target, valid_mask)
        self.replay_buffer: deque = deque(maxlen=buffer_capacity)

        # Training state
        self.step_count = 0
        self.hidden_layers = hidden_layers

    def select_action(
        self,
        state: Tensor,
        valid_mask: Tensor,
        training: bool = True
    ) -> Tuple[int, Tensor]:
        """Select action using MCTS.

        Args:
            state: (16, 17) current board state (single game)
            valid_mask: (4,) valid action mask
            training: If True, use exploration; if False, deterministic

        Returns:
            Tuple of (action, policy_target)
        """
        # Determine temperature based on training step
        if training:
            if self.step_count >= self.mcts_config.temperature_drop_step:
                temperature = 0.5  # Reduced exploration later in training
            else:
                temperature = self.mcts_config.temperature
        else:
            temperature = 0.0  # Deterministic for evaluation

        # Run MCTS
        root = self.mcts.search(
            state,
            add_exploration_noise=training
        )

        # Get action and policy target
        action = self.mcts.select_action(root, temperature)
        policy_target = self.mcts.get_policy_target(root)

        return action, policy_target

    def select_action_batch(
        self,
        states: Tensor,
        valid_masks: Tensor,
        training: bool = True
    ) -> Tensor:
        """Select actions for a batch of states.

        For MCTS, we run search sequentially for each state.
        This is simpler than tree parallelism.

        Args:
            states: (N, 16, 17) current board states
            valid_masks: (N, 4) valid action masks
            training: If True, use exploration

        Returns:
            (N,) selected actions
        """
        batch_size = states.size(0)
        actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            action, _ = self.select_action(
                states[i], valid_masks[i], training
            )
            actions[i] = action

        return actions

    def store_experience(
        self,
        state: Tensor,
        policy_target: Tensor,
        value_target: float,
        valid_mask: Tensor
    ) -> None:
        """Store training experience.

        Args:
            state: (16, 17) game state
            policy_target: (4,) MCTS visit count distribution
            value_target: Actual return from this state
            valid_mask: (4,) valid actions mask
        """
        self.replay_buffer.append({
            'state': state.clone(),
            'policy_target': policy_target.clone(),
            'value_target': value_target,
            'valid_mask': valid_mask.clone(),
        })

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step.

        Uses cross-entropy for policy loss and MSE for value loss.

        Returns:
            Dict with training metrics or None if buffer not ready
        """
        if len(self.replay_buffer) < self.buffer_min_size:
            return None

        # Sample batch
        indices = torch.randint(
            len(self.replay_buffer), (self.batch_size,)
        )

        states = []
        policy_targets = []
        value_targets = []
        valid_masks = []

        for idx in indices:
            exp = self.replay_buffer[idx]
            states.append(exp['state'])
            policy_targets.append(exp['policy_target'])
            value_targets.append(exp['value_target'])
            valid_masks.append(exp['valid_mask'])

        states = torch.stack(states).to(self.device)
        policy_targets = torch.stack(policy_targets).to(self.device)
        value_targets = torch.tensor(value_targets, device=self.device)
        valid_masks = torch.stack(valid_masks).to(self.device)

        # Forward pass
        repr_states = self.representation(states)
        log_policy, values = self.network.get_log_policy_and_value(
            repr_states, valid_masks
        )

        # Policy loss: cross-entropy between MCTS policy and network policy
        # -sum(target * log(predicted))
        policy_loss = -(policy_targets * log_policy).sum(dim=1).mean()

        # Value loss: MSE between predicted and actual returns
        value_loss = F.mse_loss(values, value_targets)

        # Total loss
        loss = (self.policy_loss_weight * policy_loss +
                self.value_loss_weight * value_loss)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "value_mean": values.mean().item(),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "hidden_layers": self.hidden_layers,
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
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
