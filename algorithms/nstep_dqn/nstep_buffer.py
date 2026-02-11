"""
n-step Replay Buffer for n-step DQN.

Stores transitions and computes n-step returns for training.
Uses a sliding window to accumulate n-step sequences before
storing them in the main replay buffer.

R_n = r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^{n-1}*r_{n-1} + gamma^n * Q(s_n, a*)

Per DEC-0003: Replay buffers MUST NOT contain cross-episode transitions.
Per DEC-0039: All tensor operations must be vectorized.
"""

from collections import deque
from typing import Tuple, List

import torch
from torch import Tensor


class NStepReplayBuffer:
    """Replay buffer that stores transitions with n-step returns.

    Maintains a sliding window of recent transitions per game to compute
    n-step returns. Only stores complete n-step sequences (or terminal ones).

    Per DEC-0003: Episode boundaries are handled correctly - n-step sequences
    are truncated at episode end.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        n_steps: int = 3,
        gamma: float = 0.99,
    ):
        """Initialize n-step replay buffer.

        Args:
            capacity: Maximum number of n-step transitions to store
            device: Device to store tensors on
            n_steps: Number of steps for multi-step returns
            gamma: Discount factor
        """
        self.capacity = capacity
        self.device = device
        self.n_steps = n_steps
        self.gamma = gamma

        # Pre-allocate storage tensors
        self.states = torch.zeros(
            capacity, 16, 17, dtype=torch.bool, device=device
        )
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        # n-step return instead of single reward
        self.nstep_returns = torch.zeros(capacity, dtype=torch.float32, device=device)
        # State after n steps (or terminal state if episode ended early)
        self.next_states = torch.zeros(
            capacity, 16, 17, dtype=torch.bool, device=device
        )
        # Done at any point in the n-step sequence
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.valid_masks = torch.zeros(
            capacity, 4, dtype=torch.bool, device=device
        )
        # Actual number of steps (can be < n_steps if episode ended)
        self.actual_n = torch.zeros(capacity, dtype=torch.long, device=device)

        self.position = 0
        self.size = 0

        # Temporary storage for accumulating n-step sequences
        # Will be initialized on first push based on batch size
        self.n_games = None
        self.transition_buffers: List[deque] = []

    def _init_buffers(self, n_games: int):
        """Initialize per-game transition buffers."""
        self.n_games = n_games
        self.transition_buffers = [deque(maxlen=self.n_steps) for _ in range(n_games)]

    def _compute_nstep_return(self, transitions: List) -> Tuple[float, int]:
        """Compute n-step return from a list of transitions.

        R_n = sum_{i=0}^{n-1} gamma^i * r_i

        Args:
            transitions: List of (reward, done) tuples

        Returns:
            Tuple of (n-step return, actual number of steps)
        """
        nstep_return = 0.0
        actual_n = 0

        for i, (reward, done) in enumerate(transitions):
            nstep_return += (self.gamma ** i) * reward
            actual_n = i + 1
            if done:
                break

        return nstep_return, actual_n

    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        valid_mask: Tensor
    ) -> None:
        """Add a batch of transitions and compute n-step returns.

        Each game maintains its own sliding window. When a window is full
        or an episode ends, we store the n-step transition.

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards received
            next_state: (N, 16, 17) next states
            done: (N,) episode termination flags
            valid_mask: (N, 4) valid actions for next states
        """
        batch_size = state.size(0)

        # Initialize buffers on first call
        if self.n_games is None:
            self._init_buffers(batch_size)

        # Process each game in the batch
        for i in range(batch_size):
            # Add current transition to buffer
            self.transition_buffers[i].append({
                'state': state[i].clone(),
                'action': action[i].item(),
                'reward': reward[i].item(),
                'next_state': next_state[i].clone(),
                'done': done[i].item(),
                'valid_mask': valid_mask[i].clone(),
            })

            # If episode ended, flush the buffer
            if done[i].item():
                self._flush_buffer(i, terminal=True)
            # If buffer is full, store oldest n-step transition
            elif len(self.transition_buffers[i]) == self.n_steps:
                self._store_nstep_transition(i)

    def _store_nstep_transition(self, game_idx: int) -> None:
        """Store oldest n-step transition from buffer.

        Args:
            game_idx: Index of game in batch
        """
        buffer = self.transition_buffers[game_idx]
        if len(buffer) == 0:
            return

        # Get the oldest transition's state and action
        oldest = buffer[0]
        s = oldest['state']
        a = oldest['action']

        # Compute n-step return
        transitions = [(t['reward'], t['done']) for t in buffer]
        nstep_return, actual_n = self._compute_nstep_return(transitions)

        # Get the n-step ahead state and validity
        last_idx = min(actual_n - 1, len(buffer) - 1)
        last_trans = buffer[last_idx]
        s_n = last_trans['next_state']
        done_n = last_trans['done']
        valid_n = last_trans['valid_mask']

        # Store to main buffer
        idx = self.position
        self.states[idx] = s
        self.actions[idx] = a
        self.nstep_returns[idx] = nstep_return
        self.next_states[idx] = s_n
        self.dones[idx] = done_n
        self.valid_masks[idx] = valid_n
        self.actual_n[idx] = actual_n

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _flush_buffer(self, game_idx: int, terminal: bool = False) -> None:
        """Flush all transitions from a game buffer (episode ended).

        When an episode ends, we need to store all remaining transitions
        with truncated n-step returns.

        Args:
            game_idx: Index of game in batch
            terminal: Whether this is due to episode termination
        """
        buffer = self.transition_buffers[game_idx]

        # Store each remaining transition with decreasing n-step windows
        while len(buffer) > 0:
            self._store_nstep_transition(game_idx)
            buffer.popleft()

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a random batch of n-step transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, nstep_returns, next_states, dones,
                     valid_masks, actual_n)
            Note: Returns nstep_returns instead of single rewards!
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.nstep_returns[indices],
            self.next_states[indices],
            self.dones[indices],
            self.valid_masks[indices],
            self.actual_n[indices],
        )

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size
