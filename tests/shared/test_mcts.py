"""Tests for shared MCTS infrastructure."""

import pytest
import torch
from typing import Tuple

from algorithms.shared.tree import TreeNode
from algorithms.shared.mcts_base import MCTSBase, MCTSConfig, DynamicsProvider


class MockDynamicsProvider(DynamicsProvider):
    """Mock dynamics provider for testing MCTS."""

    def __init__(self, device: torch.device):
        self.device = device
        self.step_count = 0

    def get_initial_state(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return observation as state, all actions valid."""
        valid_mask = torch.ones(4, dtype=torch.bool, device=self.device)
        return observation, valid_mask

    def step(
        self,
        state: torch.Tensor,
        action: int
    ) -> Tuple[torch.Tensor, float, bool, torch.Tensor]:
        """Mock step returning deterministic results."""
        self.step_count += 1
        next_state = state.clone()
        reward = float(action + 1)  # Reward proportional to action
        done = self.step_count >= 10  # End after 10 steps
        valid_mask = torch.ones(4, dtype=torch.bool, device=self.device)
        return next_state, reward, done, valid_mask

    def get_policy_value(
        self,
        state: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Return uniform policy and constant value."""
        # Uniform policy over valid actions
        policy = valid_mask.float()
        policy = policy / policy.sum()
        value = 1.0
        return policy, value


class TestTreeNode:
    """Test TreeNode class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = TreeNode(prior=0.5)

        assert node.prior == 0.5
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.parent is None
        assert len(node.children) == 0

    def test_node_value(self):
        """Test value computation."""
        node = TreeNode()
        node.visit_count = 10
        node.value_sum = 50.0

        assert node.value == 5.0

    def test_node_value_zero_visits(self):
        """Test value with zero visits returns zero."""
        node = TreeNode()
        assert node.value == 0.0

    def test_is_expanded(self):
        """Test expansion check."""
        node = TreeNode()
        assert not node.is_expanded()

        # Add a child
        node.children[0] = TreeNode()
        assert node.is_expanded()

    def test_expand(self):
        """Test node expansion."""
        node = TreeNode()

        action_priors = torch.tensor([0.25, 0.25, 0.25, 0.25])
        valid_mask = torch.tensor([True, False, True, True])

        node.expand(action_priors, valid_mask)

        # Should only create children for valid actions
        assert len(node.children) == 3
        assert 0 in node.children
        assert 1 not in node.children  # Invalid
        assert 2 in node.children
        assert 3 in node.children

    def test_select_child_puct(self):
        """Test PUCT-based child selection."""
        root = TreeNode()
        root.visit_count = 100

        # Create children with different priors and values
        for action in range(4):
            child = TreeNode(prior=0.25, parent=root, action_from_parent=action)
            child.visit_count = 10
            child.value_sum = action * 10.0  # Higher action = higher value
            root.children[action] = child

        # With high c_puct, exploration dominates
        # With low c_puct, exploitation dominates
        # Action 3 has highest value, should be selected with low exploration
        action, child = root.select_child(c_puct=0.01)
        assert action == 3

    def test_backpropagate(self):
        """Test value backpropagation."""
        root = TreeNode()
        child = TreeNode(parent=root)
        grandchild = TreeNode(parent=child)

        grandchild.backpropagate(10.0)

        assert grandchild.visit_count == 1
        assert grandchild.value_sum == 10.0
        assert child.visit_count == 1
        assert child.value_sum == 10.0
        assert root.visit_count == 1
        assert root.value_sum == 10.0

    def test_select_action_by_visit_count(self):
        """Test action selection by visit count."""
        root = TreeNode()

        for action in range(4):
            child = TreeNode(parent=root, action_from_parent=action)
            child.visit_count = (action + 1) * 10  # 10, 20, 30, 40
            root.children[action] = child

        # With temperature 0, should select most visited
        action = root.select_action_by_visit_count(temperature=0)
        assert action == 3

    def test_get_action_distribution(self):
        """Test visit count distribution."""
        root = TreeNode()

        root.children[0] = TreeNode()
        root.children[0].visit_count = 10
        root.children[2] = TreeNode()
        root.children[2].visit_count = 30

        dist = root.get_action_distribution()

        assert dist.shape == (4,)
        assert dist[0] == 0.25  # 10/40
        assert dist[1] == 0.0
        assert dist[2] == 0.75  # 30/40
        assert dist[3] == 0.0


class TestMCTSBase:
    """Test MCTSBase class."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def mcts(self, device):
        """Create MCTS with mock dynamics."""
        dynamics = MockDynamicsProvider(device)
        config = MCTSConfig(
            num_simulations=10,
            c_puct=1.5,
            dirichlet_alpha=0.25,
            exploration_fraction=0.0,  # Disable noise for testing
        )
        return MCTSBase(dynamics, config, device)

    def test_search_returns_root(self, mcts, device):
        """Test search returns a root node."""
        observation = torch.zeros(16, 17, device=device)

        root = mcts.search(observation, add_exploration_noise=False)

        assert isinstance(root, TreeNode)
        assert root.is_expanded()
        assert root.visit_count > 0

    def test_search_builds_tree(self, mcts, device):
        """Test search builds a search tree."""
        observation = torch.zeros(16, 17, device=device)

        root = mcts.search(observation, add_exploration_noise=False)

        # Root should have children
        assert len(root.children) > 0

        # At least some children should have visits
        total_child_visits = sum(c.visit_count for c in root.children.values())
        assert total_child_visits > 0

    def test_select_action_valid(self, mcts, device):
        """Test action selection returns valid action."""
        observation = torch.zeros(16, 17, device=device)

        root = mcts.search(observation, add_exploration_noise=False)
        action = mcts.select_action(root, temperature=0)

        assert 0 <= action < 4
        assert action in root.children

    def test_get_policy_target(self, mcts, device):
        """Test policy target extraction."""
        observation = torch.zeros(16, 17, device=device)

        root = mcts.search(observation, add_exploration_noise=False)
        policy = mcts.get_policy_target(root)

        assert policy.shape == (4,)
        assert abs(policy.sum().item() - 1.0) < 0.01  # Should sum to ~1
        assert torch.all(policy >= 0)


class TestMCTSConfig:
    """Test MCTS configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MCTSConfig()

        assert config.num_simulations == 50
        assert config.c_puct == 1.5
        assert config.dirichlet_alpha == 0.25
        assert config.exploration_fraction == 0.25
        assert config.temperature == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MCTSConfig(
            num_simulations=100,
            c_puct=2.0,
            dirichlet_alpha=0.3,
        )

        assert config.num_simulations == 100
        assert config.c_puct == 2.0
        assert config.dirichlet_alpha == 0.3
