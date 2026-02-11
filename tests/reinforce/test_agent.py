"""Tests for REINFORCE agent."""

import pytest
import torch
import tempfile
import os

from algorithms.reinforce.agent import REINFORCEAgent


class TestAgentInitialization:
    """Test agent initialization."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_default_initialization(self, device):
        """Test agent initializes with default parameters."""
        agent = REINFORCEAgent(device=device)

        assert agent.device == device
        assert agent.gamma == 0.99
        assert agent.step_count == 0

    def test_custom_initialization(self, device):
        """Test agent initializes with custom parameters."""
        agent = REINFORCEAgent(
            device=device,
            hidden_layers=[128, 64],
            learning_rate=0.001,
            gamma=0.95,
        )

        assert agent.gamma == 0.95

    def test_policy_network_initialized(self, device):
        """Test policy network is initialized."""
        agent = REINFORCEAgent(device=device)

        assert agent.policy_net is not None


class TestActionSelection:
    """Test action selection."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_greedy_selects_valid_action(self, device):
        """Test greedy selection only picks from valid actions."""
        agent = REINFORCEAgent(device=device)

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
        agent = REINFORCEAgent(device=device)

        state = torch.zeros(100, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        # Only actions 1 and 3 valid
        valid_mask = torch.zeros(100, 4, dtype=torch.bool, device=device)
        valid_mask[:, 1] = True
        valid_mask[:, 3] = True

        for _ in range(10):
            actions = agent.select_action(state, valid_mask, training=True)
            assert torch.all((actions == 1) | (actions == 3))

    def test_stochastic_has_variety(self, device):
        """Test stochastic policy produces variety in actions."""
        agent = REINFORCEAgent(device=device)

        state = torch.zeros(1000, 16, 17, dtype=torch.bool, device=device)
        valid_mask = torch.ones(1000, 4, dtype=torch.bool, device=device)

        actions = agent.select_action(state, valid_mask, training=True)

        # Should have variety in actions
        unique_actions = torch.unique(actions)
        assert len(unique_actions) > 1


class TestTrajectoryCollection:
    """Test trajectory collection for REINFORCE."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_store_transition(self, device):
        """Test storing transitions builds trajectories."""
        agent = REINFORCEAgent(device=device)

        n_games = 4
        state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        action = torch.zeros(n_games, dtype=torch.long, device=device)
        reward = torch.ones(n_games, device=device)
        done = torch.zeros(n_games, dtype=torch.bool, device=device)

        # Store some transitions
        num_complete = agent.store_transition(state, action, reward, done)
        assert num_complete == 0  # No episodes done yet

        # Mark one game as done
        done[0] = True
        num_complete = agent.store_transition(state, action, reward, done)
        assert num_complete == 1  # One trajectory completed

    def test_completed_trajectories_cleared_after_training(self, device):
        """Test completed trajectories are cleared after training."""
        agent = REINFORCEAgent(device=device)

        n_games = 4
        state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True  # Set valid state
        action = torch.zeros(n_games, dtype=torch.long, device=device)
        reward = torch.ones(n_games, device=device)
        done = torch.ones(n_games, dtype=torch.bool, device=device)  # All done

        # Store transitions
        agent.store_transition(state, action, reward, done)
        assert len(agent._completed_trajectories) > 0

        # Train
        agent.train_step(min_trajectories=1)

        # Completed trajectories should be cleared
        assert len(agent._completed_trajectories) == 0


class TestReturnsComputation:
    """Test Monte Carlo returns computation."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_compute_returns_single_step(self, device):
        """Test returns for single-step trajectory."""
        agent = REINFORCEAgent(device=device, gamma=0.99)

        # Single step with reward 10
        trajectory = [(torch.zeros(16, 17, device=device), 0, 10.0)]
        returns = agent._compute_returns(trajectory)

        assert len(returns) == 1
        assert returns[0] == 10.0

    def test_compute_returns_multi_step(self, device):
        """Test returns with discounting."""
        agent = REINFORCEAgent(device=device, gamma=0.5)

        # Three steps with rewards [1, 2, 4]
        trajectory = [
            (torch.zeros(16, 17, device=device), 0, 1.0),
            (torch.zeros(16, 17, device=device), 0, 2.0),
            (torch.zeros(16, 17, device=device), 0, 4.0),
        ]
        returns = agent._compute_returns(trajectory)

        # G_2 = 4
        # G_1 = 2 + 0.5 * 4 = 4
        # G_0 = 1 + 0.5 * 4 = 3
        assert len(returns) == 3
        assert abs(returns[2] - 4.0) < 0.01
        assert abs(returns[1] - 4.0) < 0.01
        assert abs(returns[0] - 3.0) < 0.01


class TestTraining:
    """Test training step functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step_before_ready(self, device):
        """Test train_step returns None without complete trajectories."""
        agent = REINFORCEAgent(device=device)

        result = agent.train_step(min_trajectories=1)
        assert result is None

    def test_train_step_returns_metrics(self, device):
        """Test train_step returns metrics when trajectories available."""
        agent = REINFORCEAgent(device=device)

        n_games = 4
        state = torch.zeros(n_games, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True
        action = torch.zeros(n_games, dtype=torch.long, device=device)
        reward = torch.ones(n_games, device=device)
        done = torch.ones(n_games, dtype=torch.bool, device=device)

        # Store completed episode
        agent.store_transition(state, action, reward, done)

        result = agent.train_step(min_trajectories=1)

        assert result is not None
        assert "loss" in result
        assert "avg_return" in result
        assert "num_transitions" in result


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_save_and_load_checkpoint(self, device):
        """Test saving and loading checkpoint restores agent state."""
        agent1 = REINFORCEAgent(device=device, hidden_layers=[128, 64])

        # Modify agent state
        agent1.step_count = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent1.save_checkpoint(path)

            # Create new agent and load
            agent2 = REINFORCEAgent(device=device, hidden_layers=[128, 64])
            agent2.load_checkpoint(path)

            assert agent2.step_count == 1000

            # Weights should match
            for k in agent1.policy_net.state_dict():
                assert torch.equal(
                    agent1.policy_net.state_dict()[k],
                    agent2.policy_net.state_dict()[k]
                )
