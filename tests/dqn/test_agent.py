"""Tests for DQN agent."""

import pytest
import torch
import tempfile
import os

from algorithms.dqn.agent import DQNAgent


class TestAgentInitialization:
    """Test agent initialization."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_default_initialization(self, device):
        """Test agent initializes with default parameters."""
        agent = DQNAgent(device=device)

        assert agent.device == device
        assert agent.gamma == 0.99
        assert agent.epsilon_start == 1.0
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay_steps == 100000
        assert agent.step_count == 0
        assert agent.epsilon == 1.0

    def test_custom_initialization(self, device):
        """Test agent initializes with custom parameters."""
        agent = DQNAgent(
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
        agent = DQNAgent(device=device)

        # Both networks should exist and have same architecture
        assert agent.policy_net is not None
        assert agent.target_net is not None

        # Target network should be in eval mode
        assert not agent.target_net.training

    def test_target_network_matches_policy(self, device):
        """Test target network starts with same weights as policy."""
        agent = DQNAgent(device=device)

        policy_params = dict(agent.policy_net.named_parameters())
        target_params = dict(agent.target_net.named_parameters())

        for name in policy_params:
            assert torch.equal(policy_params[name], target_params[name])


class TestEpsilonSchedule:
    """Test epsilon schedule (DEC-0035: linear decay)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_epsilon_starts_at_start(self, device):
        """Test epsilon starts at epsilon_start."""
        agent = DQNAgent(device=device, epsilon_start=1.0, epsilon_end=0.01)
        assert agent._compute_epsilon() == 1.0

    def test_epsilon_linear_decay_midpoint(self, device):
        """Test epsilon decays linearly."""
        agent = DQNAgent(
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
        agent = DQNAgent(
            device=device,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=100
        )

        agent.step_count = 100
        eps = agent._compute_epsilon()

        assert eps == 0.01

    def test_epsilon_stays_at_end(self, device):
        """Test epsilon doesn't go below epsilon_end."""
        agent = DQNAgent(
            device=device,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=100
        )

        agent.step_count = 200  # Past decay
        eps = agent._compute_epsilon()

        assert eps == 0.01


class TestActionSelection:
    """Test action selection (DEC-0034: mask-based)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_greedy_selects_valid_action(self, device):
        """Test greedy selection only picks from valid actions."""
        agent = DQNAgent(device=device)
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
        agent = DQNAgent(device=device)
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

    def test_mixed_epsilon_greedy(self, device):
        """Test epsilon-greedy mixes exploration and exploitation."""
        agent = DQNAgent(device=device)
        agent.epsilon = 0.5  # 50% random

        state = torch.zeros(1000, 16, 17, dtype=torch.bool, device=device)
        valid_mask = torch.ones(1000, 4, dtype=torch.bool, device=device)

        actions = agent.select_action(state, valid_mask, training=True)

        # Should have variety in actions (not all same)
        unique_actions = torch.unique(actions)
        assert len(unique_actions) > 1


class TestTraining:
    """Test training step functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step_before_ready(self, device):
        """Test train_step returns None before buffer is ready."""
        agent = DQNAgent(device=device, buffer_min_size=100)

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
        agent = DQNAgent(device=device, buffer_min_size=10, batch_size=8)

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

    def test_train_step_updates_step_count(self, device):
        """Test train_step increments step count."""
        agent = DQNAgent(device=device, buffer_min_size=10, batch_size=8)

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

        initial_count = agent.step_count
        agent.train_step()
        assert agent.step_count == initial_count + 1


class TestTargetNetworkUpdate:
    """Test target network updates (DEC-0036: hard update)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_target_not_updated_before_frequency(self, device):
        """Test target network not updated before update_frequency steps."""
        agent = DQNAgent(
            device=device,
            target_update_frequency=10,
            buffer_min_size=10,
            batch_size=8
        )

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

        # Save initial target weights
        initial_target = {k: v.clone() for k, v in agent.target_net.state_dict().items()}

        # Train for fewer than update_frequency steps
        for _ in range(5):
            agent.train_step()

        # Target should have same weights
        for k, v in agent.target_net.state_dict().items():
            assert torch.equal(v, initial_target[k])

    def test_target_updated_at_frequency(self, device):
        """Test target network is updated at update_frequency."""
        agent = DQNAgent(
            device=device,
            target_update_frequency=5,
            buffer_min_size=10,
            batch_size=8
        )

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

        # Train for exactly update_frequency steps
        for _ in range(5):
            agent.train_step()

        # After update, target should match current policy
        policy_state = agent.policy_net.state_dict()
        target_state = agent.target_net.state_dict()

        for k in policy_state:
            assert torch.equal(policy_state[k], target_state[k])


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_save_and_load_checkpoint(self, device):
        """Test saving and loading checkpoint restores agent state."""
        agent1 = DQNAgent(device=device, hidden_layers=[128, 64])

        # Modify agent state
        agent1.step_count = 1000
        agent1.epsilon = 0.5

        # Train a bit to modify weights
        for _ in range(5):
            agent1.store_transition(
                state=torch.randn(4, 16, 17, device=device),
                action=torch.randint(0, 4, (4,), device=device),
                reward=torch.randn(4, device=device),
                next_state=torch.randn(4, 16, 17, device=device),
                done=torch.zeros(4, dtype=torch.bool, device=device),
                valid_mask=torch.ones(4, 4, dtype=torch.bool, device=device),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent1.save_checkpoint(path)

            # Create new agent and load
            agent2 = DQNAgent(device=device, hidden_layers=[128, 64])
            agent2.load_checkpoint(path)

            assert agent2.step_count == 1000
            assert agent2.epsilon == 0.5

            # Weights should match
            for k in agent1.policy_net.state_dict():
                assert torch.equal(
                    agent1.policy_net.state_dict()[k],
                    agent2.policy_net.state_dict()[k]
                )

    def test_checkpoint_stores_architecture_info(self, device):
        """Test checkpoint stores hidden layer info."""
        agent = DQNAgent(device=device, hidden_layers=[128, 64, 32])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent.save_checkpoint(path)

            checkpoint = torch.load(path)
            assert "hidden_layers" in checkpoint
            assert checkpoint["hidden_layers"] == [128, 64, 32]
