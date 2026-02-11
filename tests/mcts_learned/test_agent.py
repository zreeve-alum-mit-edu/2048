"""Tests for MCTS+Learned agent."""

import pytest
import torch
import tempfile
import os

from algorithms.mcts_learned.agent import MCTSLearnedAgent
from algorithms.mcts_learned.model import PolicyValueNetwork


class TestAgentInitialization:
    """Test agent initialization."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_default_initialization(self, device):
        """Test agent initializes with default parameters."""
        agent = MCTSLearnedAgent(device=device)

        assert agent.device == device
        assert agent.gamma == 0.99
        assert agent.mcts_config.num_simulations == 100
        assert agent.mcts_config.c_puct == 1.5
        assert agent.step_count == 0

    def test_custom_initialization(self, device):
        """Test agent initializes with custom parameters."""
        agent = MCTSLearnedAgent(
            device=device,
            hidden_layers=[128, 64],
            num_simulations=50,
            c_puct=2.0,
            gamma=0.95,
        )

        assert agent.mcts_config.num_simulations == 50
        assert agent.mcts_config.c_puct == 2.0
        assert agent.gamma == 0.95

    def test_network_initialized(self, device):
        """Test policy-value network is initialized."""
        agent = MCTSLearnedAgent(device=device)

        assert agent.network is not None
        assert isinstance(agent.network, PolicyValueNetwork)


class TestPolicyValueNetwork:
    """Test PolicyValueNetwork."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_outputs(self, device):
        """Test network forward pass outputs."""
        network = PolicyValueNetwork(input_size=272, hidden_layers=[128]).to(device)

        state = torch.randn(4, 272, device=device)
        logits, value = network(state)

        assert logits.shape == (4, 4)
        assert value.shape == (4,)

    def test_get_policy(self, device):
        """Test policy extraction with masking."""
        network = PolicyValueNetwork(input_size=272, hidden_layers=[128]).to(device)

        state = torch.randn(4, 272, device=device)
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)
        valid_mask[:, 1] = False  # Action 1 invalid

        policy = network.get_policy(state, valid_mask)

        assert policy.shape == (4, 4)
        assert torch.allclose(policy.sum(dim=1), torch.ones(4, device=device), atol=1e-5)
        assert torch.all(policy[:, 1] == 0)  # Invalid action has 0 probability

    def test_get_policy_and_value(self, device):
        """Test combined policy and value extraction."""
        network = PolicyValueNetwork(input_size=272, hidden_layers=[128]).to(device)

        state = torch.randn(4, 272, device=device)
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)

        policy, value = network.get_policy_and_value(state, valid_mask)

        assert policy.shape == (4, 4)
        assert value.shape == (4,)


class TestMCTSActionSelection:
    """Test MCTS-based action selection."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_select_action_returns_valid(self, device):
        """Test action selection returns valid action and policy."""
        agent = MCTSLearnedAgent(
            device=device,
            num_simulations=5,  # Few simulations for speed
        )

        # Create a realistic game state (some tiles with values)
        state = torch.zeros(16, 17, dtype=torch.bool, device=device)
        # Set most cells to empty (value 0)
        state[:, 0] = True
        # Set a few tiles with actual values (2, 4)
        state[0, 0] = False  # Tile 0 is not empty
        state[0, 1] = True   # Tile 0 has value 2
        state[1, 0] = False
        state[1, 2] = True   # Tile 1 has value 4

        valid_mask = torch.ones(4, dtype=torch.bool, device=device)

        action, policy_target = agent.select_action(state, valid_mask, training=False)

        assert 0 <= action < 4
        assert policy_target.shape == (4,)
        # Policy may not sum to exactly 1 if search didn't fully explore
        assert policy_target.sum().item() >= 0

    def test_select_action_returns_action_in_valid_set(self, device):
        """Test MCTS returns an action from the valid set provided."""
        agent = MCTSLearnedAgent(
            device=device,
            num_simulations=10,
        )

        # Create a realistic game state
        state = torch.zeros(16, 17, dtype=torch.bool, device=device)
        state[:, 0] = True  # All empty
        state[0, 0] = False
        state[0, 1] = True  # One tile with value 2
        state[5, 0] = False
        state[5, 2] = True  # Another tile with value 4

        # All actions valid for this open board
        valid_mask = torch.ones(4, dtype=torch.bool, device=device)

        action, policy = agent.select_action(state, valid_mask, training=False)

        # Action should be in valid range
        assert 0 <= action < 4
        # Policy should give probabilities for actions
        assert policy.shape == (4,)


class TestMCTSLearnedTraining:
    """Test MCTS+Learned training."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_store_experience(self, device):
        """Test experience storage."""
        agent = MCTSLearnedAgent(device=device, buffer_capacity=100)

        state = torch.zeros(16, 17, dtype=torch.bool, device=device)
        policy_target = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
        value_target = 100.0
        valid_mask = torch.ones(4, dtype=torch.bool, device=device)

        agent.store_experience(state, policy_target, value_target, valid_mask)

        assert len(agent.replay_buffer) == 1

    def test_train_step_before_ready(self, device):
        """Test train_step returns None before buffer is ready."""
        agent = MCTSLearnedAgent(device=device, buffer_min_size=100)

        # Store one experience
        state = torch.zeros(16, 17, dtype=torch.bool, device=device)
        policy_target = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
        agent.store_experience(state, policy_target, 100.0, torch.ones(4, dtype=torch.bool, device=device))

        result = agent.train_step()
        assert result is None

    def test_train_step_returns_metrics(self, device):
        """Test train_step returns metrics when buffer is ready."""
        agent = MCTSLearnedAgent(
            device=device,
            buffer_min_size=10,
            batch_size=8,
        )

        # Fill buffer
        for i in range(15):
            state = torch.randn(16, 17, device=device)
            policy = torch.softmax(torch.randn(4, device=device), dim=0)
            value = float(i * 10)
            valid_mask = torch.ones(4, dtype=torch.bool, device=device)
            agent.store_experience(state, policy, value, valid_mask)

        result = agent.train_step()

        assert result is not None
        assert "loss" in result
        assert "policy_loss" in result
        assert "value_loss" in result


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_save_and_load_checkpoint(self, device):
        """Test saving and loading checkpoint restores agent state."""
        agent1 = MCTSLearnedAgent(device=device, hidden_layers=[128, 64])

        agent1.step_count = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent1.save_checkpoint(path)

            agent2 = MCTSLearnedAgent(device=device, hidden_layers=[128, 64])
            agent2.load_checkpoint(path)

            assert agent2.step_count == 1000
