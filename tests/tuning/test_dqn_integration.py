"""
Tests for DQN agent integration with all 5 representations.

Per DEC-0037: All 5 representations work with DQN.
Per AC-6: DQNNetwork accepts variable input_size.
Per AC-7: All 5 representations work with DQN.
"""

import pytest
import torch

from algorithms.dqn.agent import DQNAgent
from tuning.utils import create_representation
from tuning.search_spaces import get_default_params


class TestDQNWithRepresentations:
    """Test DQN agent works with all 5 representations."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_state(self, device):
        """Create sample state for testing."""
        batch_size = 4
        state = torch.zeros(batch_size, 16, 17, dtype=torch.bool, device=device)
        for i in range(16):
            state[:, i, 0] = True
        return state

    @pytest.fixture
    def valid_mask(self, device):
        """Create valid action mask."""
        return torch.ones(4, 4, dtype=torch.bool, device=device)

    @pytest.mark.parametrize("repr_type", [
        "onehot",
        "embedding",
        "cnn_2x2",
        "cnn_4x1",
        "cnn_multi",
    ])
    def test_agent_initializes_with_representation(self, repr_type, device):
        """Test DQN agent can initialize with each representation type."""
        params = get_default_params(repr_type)
        representation = create_representation(repr_type, params).to(device)

        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
        )

        assert agent is not None
        assert agent.representation is not None

    @pytest.mark.parametrize("repr_type", [
        "onehot",
        "embedding",
        "cnn_2x2",
        "cnn_4x1",
        "cnn_multi",
    ])
    def test_agent_select_action_works(self, repr_type, device, sample_state, valid_mask):
        """Test action selection works with each representation."""
        params = get_default_params(repr_type)
        representation = create_representation(repr_type, params).to(device)

        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=params["hidden_layers"],
        )

        actions = agent.select_action(sample_state, valid_mask, training=True)

        assert actions.shape == (4,)
        assert actions.dtype == torch.long
        assert torch.all(actions >= 0)
        assert torch.all(actions < 4)

    @pytest.mark.parametrize("repr_type", [
        "onehot",
        "embedding",
        "cnn_2x2",
        "cnn_4x1",
        "cnn_multi",
    ])
    def test_agent_greedy_action_works(self, repr_type, device, sample_state, valid_mask):
        """Test greedy action selection (training=False) works."""
        params = get_default_params(repr_type)
        representation = create_representation(repr_type, params).to(device)

        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=params["hidden_layers"],
        )

        actions = agent.select_action(sample_state, valid_mask, training=False)

        assert actions.shape == (4,)

    @pytest.mark.parametrize("repr_type", [
        "onehot",
        "embedding",
        "cnn_2x2",
        "cnn_4x1",
        "cnn_multi",
    ])
    def test_agent_store_transition_works(self, repr_type, device, sample_state, valid_mask):
        """Test storing transitions works with each representation."""
        params = get_default_params(repr_type)
        representation = create_representation(repr_type, params).to(device)

        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=params["hidden_layers"],
        )

        # Store a transition
        agent.store_transition(
            state=sample_state,
            action=torch.zeros(4, dtype=torch.long, device=device),
            reward=torch.ones(4, device=device),
            next_state=sample_state,
            done=torch.zeros(4, dtype=torch.bool, device=device),
            valid_mask=valid_mask,
        )

        assert len(agent.replay_buffer) == 4

    @pytest.mark.parametrize("repr_type", [
        "onehot",
        "embedding",
        "cnn_2x2",
        "cnn_4x1",
        "cnn_multi",
    ])
    def test_agent_train_step_works(self, repr_type, device, sample_state, valid_mask):
        """Test training step works with each representation."""
        params = get_default_params(repr_type)
        representation = create_representation(repr_type, params).to(device)

        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=params["hidden_layers"],
            buffer_min_size=10,
            batch_size=8,
        )

        # Fill buffer with enough samples
        for _ in range(5):
            agent.store_transition(
                state=sample_state,
                action=torch.randint(0, 4, (4,), device=device),
                reward=torch.randn(4, device=device),
                next_state=sample_state,
                done=torch.zeros(4, dtype=torch.bool, device=device),
                valid_mask=valid_mask,
            )

        # Train step should work
        result = agent.train_step()

        assert result is not None
        assert "loss" in result
        assert "q_mean" in result


class TestDQNNetworkVariableInputSize:
    """Test DQNNetwork accepts variable input sizes (AC-6)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_different_input_sizes_create_different_networks(self, device):
        """Test different representations create networks with different input sizes."""
        repr_types = ["onehot", "embedding", "cnn_2x2", "cnn_4x1", "cnn_multi"]
        input_sizes = []

        for repr_type in repr_types:
            params = get_default_params(repr_type)
            representation = create_representation(repr_type, params).to(device)
            input_size = representation.output_shape()[0]
            input_sizes.append(input_size)

            # Create agent - should work with any input size
            agent = DQNAgent(
                device=device,
                representation=representation,
                hidden_layers=[64],
            )
            assert agent is not None

        # At least some input sizes should differ
        assert len(set(input_sizes)) > 1

    def test_agent_works_with_small_hidden_layers(self, device):
        """Test agent works with minimal hidden layers."""
        representation = create_representation("onehot", {}).to(device)

        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=[32],  # Very small
        )

        state = torch.zeros(2, 16, 17, dtype=torch.bool, device=device)
        valid_mask = torch.ones(2, 4, dtype=torch.bool, device=device)

        actions = agent.select_action(state, valid_mask, training=True)
        assert actions.shape == (2,)

    def test_agent_works_with_deep_hidden_layers(self, device):
        """Test agent works with deep hidden layers."""
        representation = create_representation("onehot", {}).to(device)

        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=[256, 256, 128],  # 3 layers
        )

        state = torch.zeros(2, 16, 17, dtype=torch.bool, device=device)
        valid_mask = torch.ones(2, 4, dtype=torch.bool, device=device)

        actions = agent.select_action(state, valid_mask, training=True)
        assert actions.shape == (2,)


class TestRewardTypes:
    """Test both reward types work (AC-8)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_merge_reward_accepted(self, device):
        """Test merge reward can be used for training."""
        representation = create_representation("onehot", {}).to(device)
        agent = DQNAgent(device=device, representation=representation, buffer_min_size=10, batch_size=8)

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)

        # Simulate merge rewards (typically small integers)
        merge_reward = torch.tensor([0.0, 4.0, 8.0, 0.0], device=device)

        for _ in range(5):
            agent.store_transition(
                state=state,
                action=torch.randint(0, 4, (4,), device=device),
                reward=merge_reward,
                next_state=state,
                done=torch.zeros(4, dtype=torch.bool, device=device),
                valid_mask=valid_mask,
            )

        result = agent.train_step()
        assert result is not None

    def test_spawn_reward_accepted(self, device):
        """Test spawn reward can be used for training."""
        representation = create_representation("onehot", {}).to(device)
        agent = DQNAgent(device=device, representation=representation, buffer_min_size=10, batch_size=8)

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)

        # Simulate spawn rewards (typically 2 or 4)
        spawn_reward = torch.tensor([2.0, 2.0, 4.0, 2.0], device=device)

        for _ in range(5):
            agent.store_transition(
                state=state,
                action=torch.randint(0, 4, (4,), device=device),
                reward=spawn_reward,
                next_state=state,
                done=torch.zeros(4, dtype=torch.bool, device=device),
                valid_mask=valid_mask,
            )

        result = agent.train_step()
        assert result is not None


class TestBackwardCompatibility:
    """Test backward compatibility when no representation is provided."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_agent_defaults_to_onehot(self, device):
        """Test agent defaults to OneHotRepresentation when none provided."""
        from representations.onehot import OneHotRepresentation

        agent = DQNAgent(device=device)

        assert isinstance(agent.representation, OneHotRepresentation)

    def test_agent_works_without_representation_param(self, device):
        """Test agent works when representation is not provided."""
        agent = DQNAgent(device=device)

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)

        actions = agent.select_action(state, valid_mask, training=True)
        assert actions.shape == (4,)
