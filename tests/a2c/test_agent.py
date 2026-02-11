"""Tests for A2C agent."""

import pytest
import torch
import tempfile
import os

from algorithms.a2c.agent import A2CAgent


class TestAgentInitialization:
    """Test agent initialization."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_default_initialization(self, device):
        """Test agent initializes with default parameters."""
        agent = A2CAgent(device=device)

        assert agent.device == device
        assert agent.gamma == 0.99
        assert agent.n_steps == 5
        assert agent.value_loss_coef == 0.5
        assert agent.entropy_coef == 0.01
        assert agent.step_count == 0

    def test_custom_initialization(self, device):
        """Test agent initializes with custom parameters."""
        agent = A2CAgent(
            device=device,
            hidden_layers=[128, 64],
            learning_rate=0.001,
            gamma=0.95,
            n_steps=10,
            value_loss_coef=0.25,
            entropy_coef=0.05,
        )

        assert agent.gamma == 0.95
        assert agent.n_steps == 10
        assert agent.value_loss_coef == 0.25
        assert agent.entropy_coef == 0.05

    def test_network_initialized(self, device):
        """Test actor-critic network is initialized."""
        agent = A2CAgent(device=device)

        assert agent.network is not None
        # Network should have actor and critic heads
        assert hasattr(agent.network, 'actor_head')
        assert hasattr(agent.network, 'critic_head')


class TestActionSelection:
    """Test action selection."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_greedy_selects_valid_action(self, device):
        """Test greedy selection only picks from valid actions."""
        agent = A2CAgent(device=device)

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        # Only action 2 valid for all games
        valid_mask = torch.zeros(4, 4, dtype=torch.bool, device=device)
        valid_mask[:, 2] = True

        for _ in range(10):
            actions = agent.select_action(state, valid_mask, training=False)
            assert torch.all(actions == 2)

    def test_stochastic_selects_valid_action(self, device):
        """Test stochastic sampling only picks from valid actions."""
        agent = A2CAgent(device=device)

        state = torch.zeros(100, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        # Only actions 1 and 3 valid
        valid_mask = torch.zeros(100, 4, dtype=torch.bool, device=device)
        valid_mask[:, 1] = True
        valid_mask[:, 3] = True

        for _ in range(10):
            actions = agent.select_action(state, valid_mask, training=True)
            assert torch.all((actions == 1) | (actions == 3))


class TestRolloutCollection:
    """Test n-step rollout collection for A2C."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_store_transition(self, device):
        """Test storing transitions builds rollout."""
        agent = A2CAgent(device=device, n_steps=5)

        n_games = 4
        state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        action = torch.zeros(n_games, dtype=torch.long, device=device)
        reward = torch.ones(n_games, device=device)
        done = torch.zeros(n_games, dtype=torch.bool, device=device)
        valid_mask = torch.ones(n_games, 4, dtype=torch.bool, device=device)

        # Store transitions
        for _ in range(3):
            agent.store_transition(state, action, reward, done, valid_mask)

        assert len(agent._states) == 3
        assert len(agent._actions) == 3
        assert len(agent._rewards) == 3

    def test_should_update_after_n_steps(self, device):
        """Test should_update returns True after n_steps."""
        agent = A2CAgent(device=device, n_steps=5)

        n_games = 4
        state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        action = torch.zeros(n_games, dtype=torch.long, device=device)
        reward = torch.ones(n_games, device=device)
        done = torch.zeros(n_games, dtype=torch.bool, device=device)
        valid_mask = torch.ones(n_games, 4, dtype=torch.bool, device=device)

        # Not ready yet
        for _ in range(4):
            agent.store_transition(state, action, reward, done, valid_mask)
        assert not agent.should_update()

        # Now ready
        agent.store_transition(state, action, reward, done, valid_mask)
        assert agent.should_update()


class TestTraining:
    """Test training step functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step_before_ready(self, device):
        """Test train_step returns None without enough steps."""
        agent = A2CAgent(device=device, n_steps=5)

        next_state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        next_done = torch.zeros(4, dtype=torch.bool, device=device)

        result = agent.train_step(next_state, next_done)
        assert result is None

    def test_train_step_returns_metrics(self, device):
        """Test train_step returns metrics when ready."""
        agent = A2CAgent(device=device, n_steps=5)

        n_games = 4
        state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True
        action = torch.zeros(n_games, dtype=torch.long, device=device)
        reward = torch.ones(n_games, device=device)
        done = torch.zeros(n_games, dtype=torch.bool, device=device)
        valid_mask = torch.ones(n_games, 4, dtype=torch.bool, device=device)

        # Fill n_steps
        for _ in range(5):
            agent.store_transition(state, action, reward, done, valid_mask)

        next_state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        next_state[:, :, 0] = True
        next_done = torch.zeros(n_games, dtype=torch.bool, device=device)

        result = agent.train_step(next_state, next_done)

        assert result is not None
        assert "loss" in result
        assert "policy_loss" in result
        assert "value_loss" in result
        assert "entropy" in result

    def test_train_step_clears_rollout(self, device):
        """Test train_step clears rollout storage."""
        agent = A2CAgent(device=device, n_steps=5)

        n_games = 4
        state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True
        action = torch.zeros(n_games, dtype=torch.long, device=device)
        reward = torch.ones(n_games, device=device)
        done = torch.zeros(n_games, dtype=torch.bool, device=device)
        valid_mask = torch.ones(n_games, 4, dtype=torch.bool, device=device)

        # Fill n_steps
        for _ in range(5):
            agent.store_transition(state, action, reward, done, valid_mask)

        next_state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        next_done = torch.zeros(n_games, dtype=torch.bool, device=device)

        agent.train_step(next_state, next_done)

        # Rollout should be cleared
        assert len(agent._states) == 0
        assert len(agent._actions) == 0


class TestValueFunction:
    """Test value function computation."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_network_outputs_value(self, device):
        """Test network outputs value estimates."""
        agent = A2CAgent(device=device)

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        repr_state = agent.representation(state)
        value = agent.network.get_value(repr_state)

        assert value.shape == (4,)


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_save_and_load_checkpoint(self, device):
        """Test saving and loading checkpoint restores agent state."""
        agent1 = A2CAgent(device=device, hidden_layers=[128, 64])

        # Modify agent state
        agent1.step_count = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent1.save_checkpoint(path)

            # Create new agent and load
            agent2 = A2CAgent(device=device, hidden_layers=[128, 64])
            agent2.load_checkpoint(path)

            assert agent2.step_count == 1000

            # Weights should match
            for k in agent1.network.state_dict():
                assert torch.equal(
                    agent1.network.state_dict()[k],
                    agent2.network.state_dict()[k]
                )
