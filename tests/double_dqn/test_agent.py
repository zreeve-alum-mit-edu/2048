"""Tests for Double DQN agent."""

import pytest
import torch
import tempfile
import os

from algorithms.double_dqn.agent import DoubleDQNAgent


class TestAgentInitialization:
    """Test agent initialization."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_default_initialization(self, device):
        """Test agent initializes with default parameters."""
        agent = DoubleDQNAgent(device=device)

        assert agent.device == device
        assert agent.gamma == 0.99
        assert agent.epsilon_start == 1.0
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay_steps == 100000
        assert agent.step_count == 0
        assert agent.epsilon == 1.0

    def test_custom_initialization(self, device):
        """Test agent initializes with custom parameters."""
        agent = DoubleDQNAgent(
            device=device,
            hidden_layers=[128, 64],
            learning_rate=0.001,
            gamma=0.95,
            epsilon_start=0.5,
            epsilon_end=0.05,
            epsilon_decay_steps=50000,
        )

        assert agent.gamma == 0.95
        assert agent.epsilon_start == 0.5
        assert agent.epsilon_end == 0.05
        assert agent.epsilon_decay_steps == 50000

    def test_networks_initialized(self, device):
        """Test policy and target networks are initialized."""
        agent = DoubleDQNAgent(device=device)

        # Both networks should exist and have same architecture
        assert agent.policy_net is not None
        assert agent.target_net is not None

        # Target network should be in eval mode
        assert not agent.target_net.training

    def test_target_network_matches_policy(self, device):
        """Test target network starts with same weights as policy."""
        agent = DoubleDQNAgent(device=device)

        policy_params = dict(agent.policy_net.named_parameters())
        target_params = dict(agent.target_net.named_parameters())

        for name in policy_params:
            assert torch.equal(policy_params[name], target_params[name])


class TestEpsilonSchedule:
    """Test epsilon schedule (following DEC-0035 pattern)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_epsilon_starts_at_start(self, device):
        """Test epsilon starts at epsilon_start."""
        agent = DoubleDQNAgent(device=device, epsilon_start=1.0, epsilon_end=0.01)
        assert agent._compute_epsilon() == 1.0

    def test_epsilon_linear_decay_midpoint(self, device):
        """Test epsilon decays linearly."""
        agent = DoubleDQNAgent(
            device=device,
            epsilon_start=1.0,
            epsilon_end=0.0,
            epsilon_decay_steps=100
        )

        agent.step_count = 50  # Halfway
        eps = agent._compute_epsilon()

        assert abs(eps - 0.5) < 0.01

    def test_epsilon_ends_at_end(self, device):
        """Test epsilon reaches epsilon_end after decay_steps."""
        agent = DoubleDQNAgent(
            device=device,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=100
        )

        agent.step_count = 100
        eps = agent._compute_epsilon()

        assert eps == 0.01


class TestActionSelection:
    """Test action selection (DEC-0034: mask-based)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_greedy_selects_valid_action(self, device):
        """Test greedy selection only picks from valid actions."""
        agent = DoubleDQNAgent(device=device)
        agent.epsilon = 0.0  # Force greedy

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        # Only action 2 valid for all games
        valid_mask = torch.zeros(4, 4, dtype=torch.bool, device=device)
        valid_mask[:, 2] = True

        for _ in range(10):  # Multiple trials
            actions = agent.select_action(state, valid_mask, training=False)
            assert torch.all(actions == 2)

    def test_exploration_selects_valid_action(self, device):
        """Test random exploration only picks from valid actions."""
        agent = DoubleDQNAgent(device=device)
        agent.epsilon = 1.0  # Force random

        state = torch.zeros(100, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        # Only actions 1 and 3 valid
        valid_mask = torch.zeros(100, 4, dtype=torch.bool, device=device)
        valid_mask[:, 1] = True
        valid_mask[:, 3] = True

        for _ in range(10):
            actions = agent.select_action(state, valid_mask, training=True)
            assert torch.all((actions == 1) | (actions == 3))


class TestDoubleDQNTraining:
    """Test Double DQN specific training behavior."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step_before_ready(self, device):
        """Test train_step returns None before buffer is ready."""
        agent = DoubleDQNAgent(device=device, buffer_min_size=100)

        # Add only a few transitions
        agent.store_transition(
            state=torch.zeros(4, 16, 17, dtype=torch.bool, device=device),
            action=torch.zeros(4, dtype=torch.long, device=device),
            reward=torch.zeros(4, dtype=torch.float32, device=device),
            next_state=torch.zeros(4, 16, 17, dtype=torch.bool, device=device),
            done=torch.zeros(4, dtype=torch.bool, device=device),
            valid_mask=torch.ones(4, 4, dtype=torch.bool, device=device),
        )

        result = agent.train_step()
        assert result is None

    def test_train_step_returns_metrics(self, device):
        """Test train_step returns metrics when buffer is ready."""
        agent = DoubleDQNAgent(device=device, buffer_min_size=10, batch_size=8)

        # Fill buffer
        for _ in range(5):
            agent.store_transition(
                state=torch.randn(4, 16, 17, device=device),
                action=torch.randint(0, 4, (4,), device=device),
                reward=torch.randn(4, device=device),
                next_state=torch.randn(4, 16, 17, device=device),
                done=torch.zeros(4, dtype=torch.bool, device=device),
                valid_mask=torch.ones(4, 4, dtype=torch.bool, device=device),
            )

        result = agent.train_step()

        assert result is not None
        assert "loss" in result
        assert "q_mean" in result
        assert "q_max" in result
        assert "epsilon" in result

    def test_double_dqn_uses_policy_for_action_selection(self, device):
        """Test Double DQN uses policy net for action selection in TD target."""
        # This is more of a structural test - the key difference is
        # that Double DQN decouples action selection from evaluation
        agent = DoubleDQNAgent(device=device, buffer_min_size=10, batch_size=8)

        # Fill buffer with transitions
        for _ in range(5):
            agent.store_transition(
                state=torch.randn(4, 16, 17, device=device),
                action=torch.randint(0, 4, (4,), device=device),
                reward=torch.randn(4, device=device),
                next_state=torch.randn(4, 16, 17, device=device),
                done=torch.zeros(4, dtype=torch.bool, device=device),
                valid_mask=torch.ones(4, 4, dtype=torch.bool, device=device),
            )

        # Should train without error
        result = agent.train_step()
        assert result is not None


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_save_and_load_checkpoint(self, device):
        """Test saving and loading checkpoint restores agent state."""
        agent1 = DoubleDQNAgent(device=device, hidden_layers=[128, 64])

        # Modify agent state
        agent1.step_count = 1000
        agent1.epsilon = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent1.save_checkpoint(path)

            # Create new agent and load
            agent2 = DoubleDQNAgent(device=device, hidden_layers=[128, 64])
            agent2.load_checkpoint(path)

            assert agent2.step_count == 1000
            assert agent2.epsilon == 0.5

            # Weights should match
            for k in agent1.policy_net.state_dict():
                assert torch.equal(
                    agent1.policy_net.state_dict()[k],
                    agent2.policy_net.state_dict()[k]
                )
