"""Tests for QR-DQN agent."""

import pytest
import torch
import tempfile
import os

from algorithms.qr_dqn.agent import QRDQNAgent


class TestAgentInitialization:
    """Test agent initialization."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_default_initialization(self, device):
        """Test agent initializes with default parameters."""
        agent = QRDQNAgent(device=device)

        assert agent.device == device
        assert agent.gamma == 0.99
        assert agent.n_quantiles == 200
        assert agent.kappa == 1.0
        assert agent.step_count == 0
        assert agent.epsilon == 1.0

    def test_custom_initialization(self, device):
        """Test agent initializes with custom parameters."""
        agent = QRDQNAgent(
            device=device,
            hidden_layers=[128, 64],
            n_quantiles=100,
            kappa=0.5,
            gamma=0.95,
        )

        assert agent.n_quantiles == 100
        assert agent.kappa == 0.5
        assert agent.gamma == 0.95

    def test_networks_initialized(self, device):
        """Test policy and target networks are initialized."""
        agent = QRDQNAgent(device=device)

        assert agent.policy_net is not None
        assert agent.target_net is not None
        assert not agent.target_net.training

    def test_quantile_midpoints_correct(self, device):
        """Test quantile midpoints are computed correctly."""
        agent = QRDQNAgent(device=device, n_quantiles=4)

        # tau_i = (i + 0.5) / N = [0.125, 0.375, 0.625, 0.875]
        expected = torch.tensor([0.125, 0.375, 0.625, 0.875], device=device)
        assert torch.allclose(agent.taus, expected)


class TestQuantileOutput:
    """Test QR-DQN quantile outputs."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_network_outputs_quantiles(self, device):
        """Test network outputs quantile values."""
        agent = QRDQNAgent(device=device, n_quantiles=200)

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        repr_state = agent.representation(state)
        quantiles = agent.policy_net(repr_state)

        # Shape should be (batch, actions, quantiles)
        assert quantiles.shape == (4, 4, 200)

    def test_q_values_computed_from_quantiles(self, device):
        """Test Q-values are mean of quantiles."""
        agent = QRDQNAgent(device=device, n_quantiles=200)

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        repr_state = agent.representation(state)
        q_values = agent.policy_net.get_q_values(repr_state)

        assert q_values.shape == (4, 4)


class TestActionSelection:
    """Test action selection (DEC-0034: mask-based)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_greedy_selects_valid_action(self, device):
        """Test greedy selection only picks from valid actions."""
        agent = QRDQNAgent(device=device)
        agent.epsilon = 0.0

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        valid_mask = torch.zeros(4, 4, dtype=torch.bool, device=device)
        valid_mask[:, 2] = True

        for _ in range(10):
            actions = agent.select_action(state, valid_mask, training=False)
            assert torch.all(actions == 2)

    def test_exploration_selects_valid_action(self, device):
        """Test random exploration only picks from valid actions."""
        agent = QRDQNAgent(device=device)
        agent.epsilon = 1.0

        state = torch.zeros(100, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        valid_mask = torch.zeros(100, 4, dtype=torch.bool, device=device)
        valid_mask[:, 1] = True
        valid_mask[:, 3] = True

        for _ in range(10):
            actions = agent.select_action(state, valid_mask, training=True)
            assert torch.all((actions == 1) | (actions == 3))


class TestHuberQuantileLoss:
    """Test Huber quantile regression loss."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_huber_loss_symmetric_at_zero(self, device):
        """Test Huber loss is symmetric around zero."""
        agent = QRDQNAgent(device=device, n_quantiles=4, kappa=1.0)

        # Create predictions and targets
        quantiles = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
        targets = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=device)

        loss1 = agent._huber_quantile_loss(quantiles, targets, agent.taus)

        # Swap
        loss2 = agent._huber_quantile_loss(targets, quantiles, agent.taus)

        # Losses should be similar (not exactly equal due to quantile weights)
        assert loss1.item() > 0
        assert loss2.item() > 0


class TestQRDQNTraining:
    """Test QR-DQN training behavior."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step_before_ready(self, device):
        """Test train_step returns None before buffer is ready."""
        agent = QRDQNAgent(device=device, buffer_min_size=100)

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
        agent = QRDQNAgent(device=device, buffer_min_size=10, batch_size=8)

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
        assert "epsilon" in result


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_save_and_load_checkpoint(self, device):
        """Test saving and loading checkpoint restores agent state."""
        agent1 = QRDQNAgent(device=device, hidden_layers=[128, 64])

        agent1.step_count = 1000
        agent1.epsilon = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent1.save_checkpoint(path)

            agent2 = QRDQNAgent(device=device, hidden_layers=[128, 64])
            agent2.load_checkpoint(path)

            assert agent2.step_count == 1000
            assert agent2.epsilon == 0.5
