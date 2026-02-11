"""Tests for MuZero-style agent."""

import pytest
import torch
import tempfile
import os

from algorithms.muzero_style.agent import MuZeroAgent
from algorithms.muzero_style.model import MuZeroNetworks, RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from algorithms.muzero_style.buffer import Trajectory, TrajectoryBuffer


class TestMuZeroNetworks:
    """Test MuZero network components."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_representation_network(self, device):
        """Test representation network."""
        network = RepresentationNetwork(
            input_size=272,
            hidden_size=64,
            hidden_layers=[128]
        ).to(device)

        obs = torch.randn(4, 272, device=device)
        hidden = network(obs)

        assert hidden.shape == (4, 64)
        # Should be normalized to [0, 1]
        assert torch.all(hidden >= 0)
        assert torch.all(hidden <= 1)

    def test_dynamics_network(self, device):
        """Test dynamics network."""
        network = DynamicsNetwork(
            hidden_size=64,
            num_actions=4,
            hidden_layers=[128]
        ).to(device)

        hidden = torch.randn(4, 64, device=device)
        action = torch.randint(0, 4, (4,), device=device)

        next_hidden, reward = network(hidden, action)

        assert next_hidden.shape == (4, 64)
        assert reward.shape == (4,)

    def test_prediction_network(self, device):
        """Test prediction network."""
        network = PredictionNetwork(
            hidden_size=64,
            num_actions=4,
            hidden_layers=[128]
        ).to(device)

        hidden = torch.randn(4, 64, device=device)
        policy_logits, value = network(hidden)

        assert policy_logits.shape == (4, 4)
        assert value.shape == (4,)

    def test_muzero_networks_initial_inference(self, device):
        """Test combined networks initial inference."""
        networks = MuZeroNetworks(
            input_size=272,
            hidden_size=64,
        ).to(device)

        obs = torch.randn(4, 272, device=device)
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)

        hidden, policy, value = networks.initial_inference(obs, valid_mask)

        assert hidden.shape == (4, 64)
        assert policy.shape == (4, 4)
        assert value.shape == (4,)
        # Policy should sum to 1
        assert torch.allclose(policy.sum(dim=1), torch.ones(4, device=device), atol=1e-5)

    def test_muzero_networks_recurrent_inference(self, device):
        """Test combined networks recurrent inference."""
        networks = MuZeroNetworks(
            input_size=272,
            hidden_size=64,
        ).to(device)

        hidden = torch.randn(4, 64, device=device)
        action = torch.randint(0, 4, (4,), device=device)
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)

        next_hidden, reward, policy, value = networks.recurrent_inference(
            hidden, action, valid_mask
        )

        assert next_hidden.shape == (4, 64)
        assert reward.shape == (4,)
        assert policy.shape == (4, 4)
        assert value.shape == (4,)


class TestTrajectoryBuffer:
    """Test trajectory buffer."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_push_trajectory(self, device):
        """Test pushing trajectory to buffer."""
        buffer = TrajectoryBuffer(capacity=10, device=device)

        trajectory = Trajectory(
            observations=[torch.zeros(16, 17, device=device) for _ in range(6)],
            actions=[0, 1, 2, 3, 0],
            rewards=[10.0, 20.0, 30.0, 40.0, 50.0],
            policies=[torch.tensor([0.25, 0.25, 0.25, 0.25]) for _ in range(5)],
            values=[100.0, 90.0, 80.0, 70.0, 60.0],
            valid_masks=[torch.ones(4, dtype=torch.bool) for _ in range(6)],
        )

        buffer.push(trajectory)

        assert len(buffer) == 1

    def test_sample_batch(self, device):
        """Test sampling from buffer."""
        buffer = TrajectoryBuffer(capacity=10, device=device)

        # Add trajectories
        for _ in range(5):
            trajectory = Trajectory(
                observations=[torch.randn(16, 17) for _ in range(11)],
                actions=[i % 4 for i in range(10)],
                rewards=[float(i) for i in range(10)],
                policies=[torch.tensor([0.25, 0.25, 0.25, 0.25]) for _ in range(10)],
                values=[float(100 - i * 10) for i in range(10)],
                valid_masks=[torch.ones(4, dtype=torch.bool) for _ in range(11)],
            )
            buffer.push(trajectory)

        # Sample
        batch = buffer.sample(batch_size=4, unroll_steps=3)

        obs, actions, values, rewards, policies, masks = batch

        assert obs.shape == (4, 16, 17)
        assert actions.shape == (4, 3)  # unroll_steps
        assert values.shape == (4, 4)  # unroll_steps + 1
        assert rewards.shape == (4, 3)  # unroll_steps
        assert policies.shape == (4, 4, 4)  # unroll_steps + 1, 4 actions
        assert masks.shape == (4, 4, 4)  # unroll_steps + 1, 4 actions

    def test_is_ready(self, device):
        """Test buffer readiness check."""
        buffer = TrajectoryBuffer(capacity=10, device=device)

        assert not buffer.is_ready(5)

        for _ in range(5):
            trajectory = Trajectory(
                observations=[torch.zeros(16, 17) for _ in range(3)],
                actions=[0, 1],
                rewards=[1.0, 2.0],
                policies=[torch.ones(4) / 4 for _ in range(2)],
                values=[10.0, 5.0],
                valid_masks=[torch.ones(4, dtype=torch.bool) for _ in range(3)],
            )
            buffer.push(trajectory)

        assert buffer.is_ready(5)


class TestMuZeroAgent:
    """Test MuZero agent."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self, device):
        """Test agent initialization."""
        agent = MuZeroAgent(device=device)

        assert agent.device == device
        assert agent.networks is not None
        assert agent.buffer is not None

    def test_select_action(self, device):
        """Test action selection."""
        agent = MuZeroAgent(
            device=device,
            num_simulations=5,  # Few simulations for speed
        )

        state = torch.zeros(16, 17, dtype=torch.bool, device=device)
        state[:, 0] = True
        valid_mask = torch.ones(4, dtype=torch.bool, device=device)

        action, policy, value = agent.select_action(state, valid_mask, training=False)

        assert 0 <= action < 4
        assert policy.shape == (4,)
        assert isinstance(value, float)

    def test_train_step_before_ready(self, device):
        """Test train_step before buffer ready."""
        agent = MuZeroAgent(device=device, buffer_capacity=100)

        result = agent.train_step()
        assert result is None

    def test_train_step_returns_metrics(self, device):
        """Test train_step returns metrics."""
        agent = MuZeroAgent(
            device=device,
            buffer_capacity=100,
            batch_size=4,
            unroll_steps=2,
        )

        # Add trajectories
        for _ in range(15):
            trajectory = Trajectory(
                observations=[torch.randn(16, 17) for _ in range(6)],
                actions=[i % 4 for i in range(5)],
                rewards=[float(i) for i in range(5)],
                policies=[torch.softmax(torch.randn(4), dim=0) for _ in range(5)],
                values=[float(50 - i * 10) for i in range(5)],
                valid_masks=[torch.ones(4, dtype=torch.bool) for _ in range(6)],
            )
            agent.store_trajectory(trajectory)

        result = agent.train_step()

        assert result is not None
        assert "loss" in result
        assert "policy_loss" in result
        assert "value_loss" in result
        assert "reward_loss" in result


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_save_and_load_checkpoint(self, device):
        """Test saving and loading checkpoint restores agent state."""
        agent1 = MuZeroAgent(device=device, hidden_size=64)

        agent1.step_count = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent1.save_checkpoint(path)

            agent2 = MuZeroAgent(device=device, hidden_size=64)
            agent2.load_checkpoint(path)

            assert agent2.step_count == 1000
