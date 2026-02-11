"""
Prioritized Experience Replay Buffer for PER DQN.

Implements proportional prioritization using a sum-tree data structure
for efficient O(log n) sampling.

Priority: p_i = |TD_error_i| + epsilon
Probability: P(i) = p_i^alpha / sum(p_j^alpha)

Per DEC-0003: Replay buffers MUST NOT contain cross-episode transitions.
Per DEC-0039: All tensor operations must be vectorized.
"""

from typing import Tuple

import torch
from torch import Tensor


class SumTree:
    """Sum-tree data structure for efficient priority sampling.

    A complete binary tree where:
    - Leaf nodes store priorities
    - Internal nodes store sum of children
    - Root stores total sum of all priorities

    Allows O(log n) sampling and O(log n) updates.
    """

    def __init__(self, capacity: int, device: torch.device):
        """Initialize sum-tree.

        Args:
            capacity: Maximum number of elements (must be power of 2 for simplicity)
            device: Device for tensor operations
        """
        self.capacity = capacity
        self.device = device

        # Tree stored as array: [internal nodes] [leaf nodes]
        # For capacity n leaves, we need n internal nodes
        # Tree indices: 0 is root, children of i are 2i+1, 2i+2
        self.tree = torch.zeros(2 * capacity - 1, device=device)

        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree.

        Args:
            idx: Leaf index (in tree array)
            change: Change in priority value
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index for a given cumulative sum value.

        Args:
            idx: Current node index
            value: Target cumulative sum

        Returns:
            Leaf index in tree array
        """
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
        """Add new priority to tree (returns data index).

        Args:
            priority: Priority value for new transition

        Returns:
            Data index (0 to capacity-1)
        """
        data_idx = self.write_idx
        tree_idx = self.write_idx + self.capacity - 1

        self.update(data_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return data_idx

    def update(self, data_idx: int, priority: float):
        """Update priority for a transition.

        Args:
            data_idx: Data index (0 to capacity-1)
            priority: New priority value
        """
        tree_idx = data_idx + self.capacity - 1

        change = priority - self.tree[tree_idx].item()
        self.tree[tree_idx] = priority

        self._propagate(tree_idx, change)

    def get(self, value: float) -> Tuple[int, float]:
        """Sample a leaf node based on cumulative sum.

        Args:
            value: Random value in [0, total())

        Returns:
            Tuple of (data_idx, priority)
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1

        return data_idx, self.tree[tree_idx].item()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer.

    Samples transitions proportionally to their TD error priority.
    Uses importance sampling weights to correct for the sampling bias.

    Per DEC-0003: done=True means next_state is terminal.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Device to store tensors on
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling exponent
            beta_frames: Frames to anneal beta from beta_start to 1.0
            epsilon: Small constant added to TD errors for stability
        """
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        # Priority tree
        self.tree = SumTree(capacity, device)

        # Pre-allocate storage tensors
        self.states = torch.zeros(
            capacity, 16, 17, dtype=torch.bool, device=device
        )
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(
            capacity, 16, 17, dtype=torch.bool, device=device
        )
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.valid_masks = torch.zeros(
            capacity, 4, dtype=torch.bool, device=device
        )

        self.position = 0
        self.size = 0
        self.frame_count = 0
        self.max_priority = 1.0  # Track max priority for new transitions

    def _compute_beta(self) -> float:
        """Compute current beta for importance sampling."""
        fraction = min(1.0, self.frame_count / self.beta_frames)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        valid_mask: Tensor
    ) -> None:
        """Add a batch of transitions with max priority.

        New transitions get max priority to ensure they're sampled at least once.

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards received
            next_state: (N, 16, 17) next states
            done: (N,) episode termination flags
            valid_mask: (N, 4) valid actions for next states
        """
        batch_size = state.size(0)

        for i in range(batch_size):
            # Add to priority tree with max priority
            data_idx = self.tree.add(self.max_priority ** self.alpha)

            # Store transition data
            self.states[data_idx] = state[i].to(torch.bool)
            self.actions[data_idx] = action[i].long()
            self.rewards[data_idx] = reward[i].float()
            self.next_states[data_idx] = next_state[i].to(torch.bool)
            self.dones[data_idx] = done[i].bool()
            self.valid_masks[data_idx] = valid_mask[i].bool()

        self.size = self.tree.size

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a prioritized batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, valid_masks,
                     indices, weights)
            - indices: for updating priorities later
            - weights: importance sampling weights
        """
        self.frame_count += 1
        beta = self._compute_beta()

        indices = []
        priorities = []

        # Stratified sampling: divide total into batch_size segments
        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            # Sample from segment [i*segment, (i+1)*segment)
            low = segment * i
            high = segment * (i + 1)
            value = torch.rand(1, device=self.device).item() * (high - low) + low

            idx, priority = self.tree.get(value)
            indices.append(idx)
            priorities.append(priority)

        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        priorities = torch.tensor(priorities, dtype=torch.float32, device=self.device)

        # Compute importance sampling weights
        # w_i = (N * P(i))^(-beta) / max(w)
        probs = priorities / total
        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()  # Normalize

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.valid_masks[indices],
            indices,
            weights,
        )

    def update_priorities(self, indices: Tensor, td_errors: Tensor) -> None:
        """Update priorities based on TD errors.

        Args:
            indices: Batch of data indices
            td_errors: Corresponding TD errors
        """
        # Priority = |TD_error| + epsilon
        priorities = (torch.abs(td_errors) + self.epsilon).cpu().numpy()

        for idx, priority in zip(indices.cpu().numpy(), priorities):
            # Update max priority
            self.max_priority = max(self.max_priority, priority)

            # Update tree with priority^alpha
            self.tree.update(int(idx), priority ** self.alpha)

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size
