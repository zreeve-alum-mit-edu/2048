"""
ACER Experience Replay Buffer.

Stores trajectories with behavior policy probabilities for importance
sampling correction. Unlike standard replay buffers, ACER needs to
store sequences (trajectories) rather than individual transitions.

Per DEC-0003: Episode boundary handling - no cross-episode transitions.
Per DEC-0039: Vectorized tensor operations.
"""

from typing import Tuple, Optional, NamedTuple
import torch
from torch import Tensor


class Trajectory(NamedTuple):
    """A trajectory of experience for ACER.

    Contains a sequence of (state, action, reward, done, valid_mask, mu)
    where mu is the behavior policy probability of the action taken.
    """
    states: Tensor        # (T, 16, 17) sequence of states
    actions: Tensor       # (T,) sequence of actions
    rewards: Tensor       # (T,) sequence of rewards
    dones: Tensor         # (T,) sequence of done flags
    valid_masks: Tensor   # (T, 4) sequence of valid action masks
    mu: Tensor            # (T,) behavior policy probabilities


class ACERReplayBuffer:
    """Replay buffer storing trajectories for ACER.

    Stores complete trajectories rather than individual transitions.
    Each trajectory includes behavior policy probabilities for
    importance sampling correction.

    The buffer stores trajectories from all parallel environments,
    organized by game index.
    """

    def __init__(
        self,
        capacity: int,
        trajectory_length: int,
        n_games: int,
        device: torch.device
    ):
        """Initialize ACER replay buffer.

        Args:
            capacity: Maximum number of trajectories to store
            trajectory_length: Length of each trajectory (n_steps)
            n_games: Number of parallel games
            device: PyTorch device
        """
        self.capacity = capacity
        self.trajectory_length = trajectory_length
        self.n_games = n_games
        self.device = device

        # Pre-allocate storage tensors
        self._states = torch.zeros(
            capacity, trajectory_length, 16, 17,
            dtype=torch.bool, device=device
        )
        self._actions = torch.zeros(
            capacity, trajectory_length,
            dtype=torch.long, device=device
        )
        self._rewards = torch.zeros(
            capacity, trajectory_length,
            dtype=torch.float32, device=device
        )
        self._dones = torch.zeros(
            capacity, trajectory_length,
            dtype=torch.bool, device=device
        )
        self._valid_masks = torch.zeros(
            capacity, trajectory_length, 4,
            dtype=torch.bool, device=device
        )
        self._mu = torch.zeros(
            capacity, trajectory_length,
            dtype=torch.float32, device=device
        )

        # Buffer state
        self._size = 0
        self._ptr = 0

    def push_trajectories(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        valid_masks: Tensor,
        mu: Tensor
    ) -> None:
        """Store trajectories from all parallel games.

        Args:
            states: (T, N, 16, 17) states for each timestep and game
            actions: (T, N) actions for each timestep and game
            rewards: (T, N) rewards for each timestep and game
            dones: (T, N) done flags for each timestep and game
            valid_masks: (T, N, 4) valid action masks
            mu: (T, N) behavior policy probabilities
        """
        T, N = actions.shape

        # Store each game's trajectory separately
        for game_idx in range(N):
            # Extract this game's trajectory
            game_states = states[:, game_idx]      # (T, 16, 17)
            game_actions = actions[:, game_idx]    # (T,)
            game_rewards = rewards[:, game_idx]    # (T,)
            game_dones = dones[:, game_idx]        # (T,)
            game_masks = valid_masks[:, game_idx]  # (T, 4)
            game_mu = mu[:, game_idx]              # (T,)

            # Store in buffer
            idx = self._ptr
            self._states[idx, :T] = game_states
            self._actions[idx, :T] = game_actions
            self._rewards[idx, :T] = game_rewards
            self._dones[idx, :T] = game_dones
            self._valid_masks[idx, :T] = game_masks
            self._mu[idx, :T] = game_mu

            # Update pointer and size
            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        """Sample a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample

        Returns:
            Tuple of:
            - states: (B, T, 16, 17) batch of state sequences
            - actions: (B, T) batch of action sequences
            - rewards: (B, T) batch of reward sequences
            - dones: (B, T) batch of done flag sequences
            - valid_masks: (B, T, 4) batch of valid mask sequences
            - mu: (B, T) batch of behavior policy probabilities
        """
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._dones[indices],
            self._valid_masks[indices],
            self._mu[indices],
        )

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough trajectories.

        Args:
            min_size: Minimum number of trajectories required

        Returns:
            True if buffer has at least min_size trajectories
        """
        return self._size >= min_size

    def __len__(self) -> int:
        """Return current number of stored trajectories."""
        return self._size
