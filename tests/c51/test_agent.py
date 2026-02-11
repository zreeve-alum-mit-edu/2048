"""Tests for C51 agent."""

import pytest
import torch
import tempfile
import os

from algorithms.c51.agent import C51Agent


class TestAgentInitialization:
    """Test agent initialization."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_default_initialization(self, device):
        """Test agent initializes with default parameters."""
        agent = C51Agent(device=device)

        assert agent.device == device
        assert agent.gamma == 0.99
        assert agent.n_atoms == 51
        assert agent.v_min == 0.0
        assert agent.v_max == 100000.0
        assert agent.step_count == 0
        assert agent.epsilon == 1.0

    def test_custom_initialization(self, device):
        """Test agent initializes with custom parameters."""
        agent = C51Agent(
            device=device,
            hidden_layers=[128, 64],
            n_atoms=31,
            v_min=-100,
            v_max=50000,
            gamma=0.95,
        )

        assert agent.n_atoms == 31
        assert agent.v_min == -100
        assert agent.v_max == 50000
        assert agent.gamma == 0.95

    def test_networks_initialized(self, device):
        """Test policy and target networks are initialized."""
        agent = C51Agent(device=device)

        assert agent.policy_net is not None
        assert agent.target_net is not None
        assert not agent.target_net.training

    def test_support_computed_correctly(self, device):
        """Test distribution support is computed correctly."""
        agent = C51Agent(
            device=device,
            n_atoms=5,
            v_min=0.0,
            v_max=4.0
        )

        expected_support = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device=device)
        assert torch.allclose(agent.support, expected_support)


class TestDistributionalOutput:
    """Test C51 distributional outputs."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_network_outputs_distribution(self, device):
        """Test network outputs valid probability distributions."""
        agent = C51Agent(device=device, n_atoms=51)

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        repr_state = agent.representation(state)
        probs = agent.policy_net(repr_state)

        # Shape should be (batch, actions, atoms)
        assert probs.shape == (4, 4, 51)

        # Probabilities should sum to 1 over atoms
        sums = probs.sum(dim=2)
        assert torch.allclose(sums, torch.ones(4, 4, device=device), atol=1e-5)

        # All probabilities should be non-negative
        assert torch.all(probs >= 0)

    def test_q_values_computed_from_distribution(self, device):
        """Test Q-values are expected values of distribution."""
        agent = C51Agent(device=device, n_atoms=51)

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
        agent = C51Agent(device=device)
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
        agent = C51Agent(device=device)
        agent.epsilon = 1.0

        state = torch.zeros(100, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        valid_mask = torch.zeros(100, 4, dtype=torch.bool, device=device)
        valid_mask[:, 1] = True
        valid_mask[:, 3] = True

        for _ in range(10):
            actions = agent.select_action(state, valid_mask, training=True)
            assert torch.all((actions == 1) | (actions == 3))


class TestC51Training:
    """Test C51 training behavior."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step_before_ready(self, device):
        """Test train_step returns None before buffer is ready."""
        agent = C51Agent(device=device, buffer_min_size=100)

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
        agent = C51Agent(device=device, buffer_min_size=10, batch_size=8)

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
        agent1 = C51Agent(device=device, hidden_layers=[128, 64])

        agent1.step_count = 1000
        agent1.epsilon = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            agent1.save_checkpoint(path)

            agent2 = C51Agent(device=device, hidden_layers=[128, 64])
            agent2.load_checkpoint(path)

            assert agent2.step_count == 1000
            assert agent2.epsilon == 0.5
