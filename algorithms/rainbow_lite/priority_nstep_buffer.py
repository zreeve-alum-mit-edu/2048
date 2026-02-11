"""
Prioritized n-step Replay Buffer for Rainbow-lite.

Combines:
- Prioritized Experience Replay (PER): sample based on TD error
- n-step returns: store multi-step transitions

Per DEC-0003: Replay buffers MUST NOT contain cross-episode transitions.
Per DEC-0039: All tensor operations must be vectorized.
"""

from collections import deque
from typing import Tuple, List

import torch
from torch import Tensor


class SumTree:
    """Sum-tree data structure for efficient priority sampling."""

    def __init__(self, capacity: int, device: torch.device):
        """Initialize sum-tree."""
        self.capacity = capacity
        self.device = device
        self.tree = torch.zeros(2 * capacity - 1, device=device)
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index for a given cumulative sum value."""
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        """Get total sum of all priorities."""
        return self.tree[0].item()

    def add(self, priority: float) -> int:
        """Add new priority to tree."""
        data_idx = self.write_idx
        tree_idx = self.write_idx + self.capacity - 1
        self.update(data_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def update(self, data_idx: int, priority: float):
        """Update priority for a transition."""
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx].item()
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, value: float) -> Tuple[int, float]:
        """Sample a leaf node based on cumulative sum."""
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return data_idx, self.tree[tree_idx].item()


class PrioritizedNStepBuffer:
    """Combined Prioritized Experience Replay with n-step returns.

    Features:
    - Computes n-step returns: R_n = sum gamma^i * r_i
    - Priority-based sampling proportional to TD error
    - Importance sampling weights for bias correction
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        n_steps: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ):
        """Initialize prioritized n-step buffer.

        Args:
            capacity: Maximum number of transitions
            device: Device for tensors
            n_steps: Number of steps for multi-step returns
            gamma: Discount factor
            alpha: Prioritization exponent (0=uniform, 1=full)
            beta_start: Initial importance sampling correction
            beta_frames: Frames to anneal beta to 1.0
            epsilon: Small constant for stability
        """
        self.capacity = capacity
        self.device = device
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        # Priority tree
        self.tree = SumTree(capacity, device)

        # Storage
        self.states = torch.zeros(capacity, 16, 17, dtype=torch.bool, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.nstep_returns = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(capacity, 16, 17, dtype=torch.bool, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.valid_masks = torch.zeros(capacity, 4, dtype=torch.bool, device=device)
        self.actual_n = torch.zeros(capacity, dtype=torch.long, device=device)

        self.position = 0
        self.size = 0
        self.frame_count = 0
        self.max_priority = 1.0

        # Per-game n-step buffers
        self.n_games = None
        self.transition_buffers: List[deque] = []

    def _init_buffers(self, n_games: int):
        """Initialize per-game transition buffers."""
        self.n_games = n_games
        self.transition_buffers = [deque(maxlen=self.n_steps) for _ in range(n_games)]

    def _compute_beta(self) -> float:
        """Compute current beta for importance sampling."""
        fraction = min(1.0, self.frame_count / self.beta_frames)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def _compute_nstep_return(self, transitions: List) -> Tuple[float, int]:
        """Compute n-step return."""
        nstep_return = 0.0
        actual_n = 0

        for i, (reward, done) in enumerate(transitions):
            nstep_return += (self.gamma ** i) * reward
            actual_n = i + 1
            if done:
                break

        return nstep_return, actual_n

    def _store_nstep_transition(self, game_idx: int) -> None:
        """Store oldest n-step transition from buffer with max priority."""
        buffer = self.transition_buffers[game_idx]
        if len(buffer) == 0:
            return

        oldest = buffer[0]
        s = oldest['state']
        a = oldest['action']

        transitions = [(t['reward'], t['done']) for t in buffer]
        nstep_return, actual_n = self._compute_nstep_return(transitions)

        last_idx = min(actual_n - 1, len(buffer) - 1)
        last_trans = buffer[last_idx]
        s_n = last_trans['next_state']
        done_n = last_trans['done']
        valid_n = last_trans['valid_mask']

        # Add to priority tree with max priority
        data_idx = self.tree.add(self.max_priority ** self.alpha)

        # Store transition data
        self.states[data_idx] = s
        self.actions[data_idx] = a
        self.nstep_returns[data_idx] = nstep_return
        self.next_states[data_idx] = s_n
        self.dones[data_idx] = done_n
        self.valid_masks[data_idx] = valid_n
        self.actual_n[data_idx] = actual_n

        self.size = self.tree.size

    def _flush_buffer(self, game_idx: int) -> None:
        """Flush all transitions from a game buffer."""
        buffer = self.transition_buffers[game_idx]
        while len(buffer) > 0:
            self._store_nstep_transition(game_idx)
            buffer.popleft()

    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        valid_mask: Tensor
    ) -> None:
        """Add a batch of transitions."""
        batch_size = state.size(0)

        if self.n_games is None:
            self._init_buffers(batch_size)

        for i in range(batch_size):
            self.transition_buffers[i].append({
                'state': state[i].clone(),
                'action': action[i].item(),
                'reward': reward[i].item(),
                'next_state': next_state[i].clone(),
                'done': done[i].item(),
                'valid_mask': valid_mask[i].clone(),
            })

            if done[i].item():
                self._flush_buffer(i)
            elif len(self.transition_buffers[i]) == self.n_steps:
                self._store_nstep_transition(i)

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        """Sample a prioritized batch of n-step transitions.

        Returns:
            Tuple of (states, actions, nstep_returns, next_states, dones,
                     valid_masks, actual_n, indices, weights)
        """
        self.frame_count += 1
        beta = self._compute_beta()

        indices = []
        priorities = []

        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = torch.rand(1, device=self.device).item() * (high - low) + low
            idx, priority = self.tree.get(value)
            indices.append(idx)
            priorities.append(priority)

        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        priorities = torch.tensor(priorities, dtype=torch.float32, device=self.device)

        probs = priorities / total
        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.nstep_returns[indices],
            self.next_states[indices],
            self.dones[indices],
            self.valid_masks[indices],
            self.actual_n[indices],
            indices,
            weights,
        )

    def update_priorities(self, indices: Tensor, td_errors: Tensor) -> None:
        """Update priorities based on TD errors."""
        priorities = (torch.abs(td_errors) + self.epsilon).cpu().numpy()

        for idx, priority in zip(indices.cpu().numpy(), priorities):
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority ** self.alpha)

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size
