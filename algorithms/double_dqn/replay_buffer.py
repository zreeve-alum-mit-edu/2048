"""
Experience Replay Buffer for Double DQN.

Reuses the same replay buffer implementation as DQN.

Per DEC-0003: Replay buffers MUST NOT contain cross-episode transitions.
When done=True, next_state is terminal; the transition is stored but
the next transition starts fresh from reset_state.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class Transition:
    """A single transition in the replay buffer.

    Attributes:
        state: (16, 17) one-hot board state
        action: int 0-3
        reward: float merge reward
        next_state: (16, 17) next state (terminal if done=True)
        done: bool whether episode ended
        valid_mask: (4,) valid actions for next_state
    """
    state: Tensor
    action: int
    reward: float
    next_state: Tensor
    done: bool
    valid_mask: Tensor


class ReplayBuffer:
    """Circular replay buffer for experience replay.

    Stores transitions and samples random minibatches for training.

    Key invariant (DEC-0003):
    - When done=True, next_state is the terminal state, NOT the reset state
    - No cross-episode transitions are stored

    The buffer stores tensors on the specified device for efficient sampling.
    """

    def __init__(self, capacity: int, device: torch.device):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Pre-allocate storage tensors
        # State: (capacity, 16, 17)
        self.states = torch.zeros(
            capacity, 16, 17, dtype=torch.bool, device=device
        )
        # Action: (capacity,)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        # Reward: (capacity,)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        # Next state: (capacity, 16, 17)
        self.next_states = torch.zeros(
            capacity, 16, 17, dtype=torch.bool, device=device
        )
        # Done: (capacity,)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        # Valid mask: (capacity, 4)
        self.valid_masks = torch.zeros(
            capacity, 4, dtype=torch.bool, device=device
        )

    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        valid_mask: Tensor
    ) -> None:
        """Add a batch of transitions to the buffer.

        This method handles batched inputs from parallel environments.
        Per DEC-0003: done=True means next_state is terminal (not reset).

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards received (merge_reward per DEC-0033)
            next_state: (N, 16, 17) next states (terminal if done)
            done: (N,) episode termination flags
            valid_mask: (N, 4) valid actions for next states
        """
        batch_size = state.size(0)

        # Vectorized batch insertion (per DEC-0039)
        # Calculate indices for the batch
        indices = (torch.arange(batch_size, device=self.device) + self.position) % self.capacity

        # Batch write all transitions (cast to correct dtypes for compatibility)
        self.states[indices] = state.to(torch.bool)
        self.actions[indices] = action.long()
        self.rewards[indices] = reward.float()
        self.next_states[indices] = next_state.to(torch.bool)
        self.dones[indices] = done.bool()
        self.valid_masks[indices] = valid_mask.bool()

        # Update position and size
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, valid_masks)
            with shapes:
            - states: (batch_size, 16, 17)
            - actions: (batch_size,)
            - rewards: (batch_size,)
            - next_states: (batch_size, 16, 17)
            - dones: (batch_size,)
            - valid_masks: (batch_size, 4)
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.valid_masks[indices],
        )

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training.

        Args:
            min_size: Minimum number of transitions required

        Returns:
            True if buffer has at least min_size transitions
        """
        return self.size >= min_size
