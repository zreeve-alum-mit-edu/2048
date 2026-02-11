"""
MCTS Tree Node Implementation.

Provides the TreeNode class used by MCTS-based algorithms
for building and traversing the search tree.

This is a shared component used by both MuZero-style and MCTS+learned.
"""

from typing import Dict, Optional, List
import torch
from torch import Tensor
import math


class TreeNode:
    """Node in the MCTS search tree.

    Each node represents a state (or hidden state for MuZero) and tracks:
    - Visit counts for each child action
    - Value estimates for each child action
    - Prior probabilities from policy network
    - Parent reference for backpropagation
    """

    def __init__(
        self,
        prior: float = 0.0,
        parent: Optional['TreeNode'] = None,
        action_from_parent: Optional[int] = None,
    ):
        """Initialize tree node.

        Args:
            prior: Prior probability P(s,a) from policy network
            parent: Parent node (None for root)
            action_from_parent: Action taken to reach this node from parent
        """
        self.prior = prior
        self.parent = parent
        self.action_from_parent = action_from_parent

        # Child nodes indexed by action
        self.children: Dict[int, 'TreeNode'] = {}

        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0

        # State information (set by expand())
        self.state: Optional[Tensor] = None  # For MCTS+learned (real states)
        self.hidden_state: Optional[Tensor] = None  # For MuZero (latent states)
        self.reward: float = 0.0  # Predicted/actual reward reaching this node

        # Valid actions mask (set during expansion)
        self.valid_actions: Optional[Tensor] = None

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Check if this node has been expanded."""
        return len(self.children) > 0

    def is_root(self) -> bool:
        """Check if this node is the root."""
        return self.parent is None

    def expand(
        self,
        action_priors: Tensor,
        valid_mask: Tensor,
        state: Optional[Tensor] = None,
        hidden_state: Optional[Tensor] = None,
        reward: float = 0.0,
    ) -> None:
        """Expand this node by adding children for valid actions.

        Args:
            action_priors: (4,) prior probabilities from policy network
            valid_mask: (4,) boolean mask of valid actions
            state: Optional real state (for MCTS+learned)
            hidden_state: Optional hidden state (for MuZero)
            reward: Reward received reaching this node
        """
        self.state = state
        self.hidden_state = hidden_state
        self.valid_actions = valid_mask
        self.reward = reward

        # Create children only for valid actions
        for action in range(4):
            if valid_mask[action].item():
                prior = action_priors[action].item()
                self.children[action] = TreeNode(
                    prior=prior,
                    parent=self,
                    action_from_parent=action,
                )

    def select_child(self, c_puct: float) -> tuple:
        """Select best child using PUCT formula.

        UCB score = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)

        Args:
            c_puct: Exploration constant

        Returns:
            Tuple of (action, child_node) with highest UCB score
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        sqrt_parent = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            # Q-value (average value)
            q_value = child.value

            # Exploration bonus
            exploration = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)

            ucb_score = q_value + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def select_action_by_visit_count(
        self,
        temperature: float = 1.0
    ) -> int:
        """Select action based on visit counts.

        Args:
            temperature: Temperature for action selection.
                         0 = deterministic (highest visit count)
                         1 = proportional to visit counts
                         >1 = more uniform

        Returns:
            Selected action
        """
        actions = list(self.children.keys())

        # Handle edge case: no children
        if len(actions) == 0:
            # Return first valid action if no tree was built
            return 0

        visit_counts = torch.tensor(
            [self.children[a].visit_count for a in actions],
            dtype=torch.float32
        )

        # Handle edge case: all zero visit counts
        if visit_counts.sum() == 0:
            # Return first action
            return actions[0]

        if temperature == 0:
            # Deterministic: pick highest visit count
            best_idx = visit_counts.argmax().item()
            return actions[best_idx]
        else:
            # Sample proportional to visit_count ^ (1/temperature)
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)

            total = visit_counts.sum()
            if total == 0:
                # Uniform if all counts are zero after power
                probs = torch.ones_like(visit_counts) / len(visit_counts)
            else:
                probs = visit_counts / total
            action_idx = torch.multinomial(probs, 1).item()
            return actions[action_idx]

    def get_action_distribution(self) -> Tensor:
        """Get action distribution based on visit counts.

        Returns:
            (4,) tensor with visit count proportions for each action
        """
        distribution = torch.zeros(4)
        total_visits = sum(child.visit_count for child in self.children.values())

        if total_visits > 0:
            for action, child in self.children.items():
                distribution[action] = child.visit_count / total_visits

        return distribution

    def backpropagate(self, value: float) -> None:
        """Backpropagate value estimate up the tree.

        Args:
            value: Value estimate from leaf evaluation
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def get_path_to_root(self) -> List['TreeNode']:
        """Get path from this node to root.

        Returns:
            List of nodes from this node to root
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path

    def __repr__(self) -> str:
        return (f"TreeNode(visits={self.visit_count}, "
                f"value={self.value:.3f}, "
                f"prior={self.prior:.3f}, "
                f"children={list(self.children.keys())})")
