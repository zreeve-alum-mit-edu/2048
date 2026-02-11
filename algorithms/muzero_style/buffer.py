"""
MuZero Trajectory Buffer.

Stores complete game trajectories for training the MuZero networks.
Unlike standard replay buffers, MuZero needs full trajectories for
unrolling dynamics during training.

Per DEC-0003: Episode boundary handling
Per DEC-0039: Vectorized operations where possible
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class Trajectory:
    """A complete game trajectory."""
    observations: List[Tensor]  # (T+1,) list of (16, 17) states
    actions: List[int]  # (T,) actions taken
    rewards: List[float]  # (T,) rewards received
    policies: List[Tensor]  # (T,) MCTS policy targets (4,)
    values: List[float]  # (T,) value targets (discounted returns)
    valid_masks: List[Tensor]  # (T+1,) valid action masks (4,)

    def __len__(self) -> int:
        return len(self.actions)


class TrajectoryBuffer:
    """Buffer for storing and sampling MuZero trajectories.

    Supports:
    - Storing complete game trajectories
    - Sampling positions with unroll_steps of future data
    - Computing n-step value targets
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        gamma: float = 0.997,
        td_steps: int = 10,
    ):
        """Initialize trajectory buffer.

        Args:
            capacity: Maximum number of trajectories to store
            device: PyTorch device
            gamma: Discount factor for value targets
            td_steps: Number of steps for TD value targets
        """
        self.capacity = capacity
        self.device = device
        self.gamma = gamma
        self.td_steps = td_steps

        self.trajectories: deque = deque(maxlen=capacity)
        self.total_positions = 0

    def push(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer.

        Args:
            trajectory: Complete game trajectory
        """
        self.trajectories.append(trajectory)
        self.total_positions += len(trajectory)

    def sample(
        self,
        batch_size: int,
        unroll_steps: int
    ) -> Tuple[Tensor, ...]:
        """Sample a batch of positions with unroll data.

        Args:
            batch_size: Number of positions to sample
            unroll_steps: Number of steps to unroll

        Returns:
            Tuple of:
            - observations: (batch, 16, 17) initial observations
            - actions: (batch, unroll_steps) actions for unrolling
            - target_values: (batch, unroll_steps + 1) value targets
            - target_rewards: (batch, unroll_steps) reward targets
            - target_policies: (batch, unroll_steps + 1, 4) policy targets
            - valid_masks: (batch, unroll_steps + 1, 4) valid action masks
        """
        # Sample random (trajectory, position) pairs
        observations = []
        actions_batch = []
        target_values = []
        target_rewards = []
        target_policies = []
        valid_masks = []

        for _ in range(batch_size):
            # Sample trajectory
            traj_idx = torch.randint(len(self.trajectories), (1,)).item()
            traj = self.trajectories[traj_idx]

            # Sample position (leaving room for unroll)
            max_pos = max(0, len(traj) - unroll_steps)
            pos = torch.randint(max_pos + 1, (1,)).item()

            # Extract data from position
            obs = traj.observations[pos].clone()
            observations.append(obs)

            # Actions for unrolling
            pos_actions = []
            for k in range(unroll_steps):
                if pos + k < len(traj):
                    pos_actions.append(traj.actions[pos + k])
                else:
                    pos_actions.append(0)  # Padding
            actions_batch.append(torch.tensor(pos_actions, dtype=torch.long))

            # Value targets (current + unroll positions)
            pos_values = []
            for k in range(unroll_steps + 1):
                if pos + k < len(traj):
                    # Compute n-step value target
                    value = self._compute_value_target(traj, pos + k)
                    pos_values.append(value)
                else:
                    pos_values.append(0.0)  # Terminal
            target_values.append(torch.tensor(pos_values))

            # Reward targets
            pos_rewards = []
            for k in range(unroll_steps):
                if pos + k < len(traj):
                    pos_rewards.append(traj.rewards[pos + k])
                else:
                    pos_rewards.append(0.0)
            target_rewards.append(torch.tensor(pos_rewards))

            # Policy targets
            pos_policies = []
            for k in range(unroll_steps + 1):
                if pos + k < len(traj):
                    pos_policies.append(traj.policies[pos + k].clone())
                else:
                    # Uniform policy for terminal
                    pos_policies.append(torch.ones(4) / 4)
            target_policies.append(torch.stack(pos_policies))

            # Valid masks
            pos_masks = []
            for k in range(unroll_steps + 1):
                if pos + k < len(traj.valid_masks):
                    pos_masks.append(traj.valid_masks[pos + k].clone())
                else:
                    pos_masks.append(torch.ones(4, dtype=torch.bool))
            valid_masks.append(torch.stack(pos_masks))

        # Stack into batches
        observations = torch.stack(observations).to(self.device)
        actions_batch = torch.stack(actions_batch).to(self.device)
        target_values = torch.stack(target_values).to(self.device)
        target_rewards = torch.stack(target_rewards).to(self.device)
        target_policies = torch.stack(target_policies).to(self.device)
        valid_masks = torch.stack(valid_masks).to(self.device)

        return (
            observations,
            actions_batch,
            target_values,
            target_rewards,
            target_policies,
            valid_masks,
        )

    def _compute_value_target(self, traj: Trajectory, pos: int) -> float:
        """Compute n-step value target.

        Args:
            traj: Trajectory
            pos: Position in trajectory

        Returns:
            Value target (n-step return)
        """
        value = 0.0
        steps = 0

        for k in range(self.td_steps):
            if pos + k >= len(traj):
                break
            value += (self.gamma ** k) * traj.rewards[pos + k]
            steps = k + 1

        # Bootstrap from stored value if not terminal
        if pos + steps < len(traj.values):
            value += (self.gamma ** steps) * traj.values[pos + steps]

        return value

    def __len__(self) -> int:
        return len(self.trajectories)

    def is_ready(self, min_trajectories: int) -> bool:
        return len(self.trajectories) >= min_trajectories
