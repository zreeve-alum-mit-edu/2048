"""Tests for DQN network model."""

import pytest
import torch

from algorithms.dqn.model import DQNNetwork


# Default input size for one-hot representation (16 * 17 = 272)
DEFAULT_INPUT_SIZE = 272


class TestDQNNetworkArchitecture:
    """Test network architecture correctness."""

    def test_default_architecture(self):
        """Test default network creates correct layer sizes."""
        net = DQNNetwork(input_size=DEFAULT_INPUT_SIZE)

        # Should have 2 hidden layers of 256 + output layer
        linear_layers = [m for m in net.network if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 3  # 2 hidden + 1 output

        assert linear_layers[0].in_features == 272  # 16 * 17
        assert linear_layers[0].out_features == 256
        assert linear_layers[1].in_features == 256
        assert linear_layers[1].out_features == 256
        assert linear_layers[2].in_features == 256
        assert linear_layers[2].out_features == 4

    def test_custom_hidden_layers(self):
        """Test custom hidden layer configuration."""
        net = DQNNetwork(input_size=DEFAULT_INPUT_SIZE, hidden_layers=[128, 64, 32])

        linear_layers = [m for m in net.network if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 4  # 3 hidden + 1 output

        assert linear_layers[0].out_features == 128
        assert linear_layers[1].out_features == 64
        assert linear_layers[2].out_features == 32
        assert linear_layers[3].out_features == 4

    def test_single_hidden_layer(self):
        """Test single hidden layer configuration."""
        net = DQNNetwork(input_size=DEFAULT_INPUT_SIZE, hidden_layers=[512])

        linear_layers = [m for m in net.network if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 2

        assert linear_layers[0].in_features == 272
        assert linear_layers[0].out_features == 512
        assert linear_layers[1].in_features == 512
        assert linear_layers[1].out_features == 4

    def test_custom_input_size(self):
        """Test network accepts different input sizes (DEC-0037)."""
        # Embedding representation with embed_dim=32: 16 * 32 = 512
        net = DQNNetwork(input_size=512, hidden_layers=[256, 256])

        linear_layers = [m for m in net.network if isinstance(m, torch.nn.Linear)]
        assert linear_layers[0].in_features == 512


class TestDQNNetworkForward:
    """Test forward pass functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def network(self, device):
        return DQNNetwork(input_size=DEFAULT_INPUT_SIZE).to(device)

    def test_forward_3d_input(self, network, device):
        """Test forward pass with (N, 16, 17) input."""
        batch_size = 8
        state = torch.zeros(batch_size, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True  # All empty boards

        output = network(state)

        assert output.shape == (batch_size, 4)
        assert output.dtype == torch.float32

    def test_forward_2d_input(self, network, device):
        """Test forward pass with pre-flattened (N, 272) input."""
        batch_size = 8
        state = torch.zeros(batch_size, 272, dtype=torch.float32, device=device)

        output = network(state)

        assert output.shape == (batch_size, 4)

    def test_forward_single_sample(self, network, device):
        """Test forward pass with single sample."""
        state = torch.zeros(1, 16, 17, dtype=torch.bool, device=device)
        state[0, :, 0] = True

        output = network(state)

        assert output.shape == (1, 4)

    def test_forward_deterministic(self, network, device):
        """Test forward pass is deterministic in eval mode."""
        network.eval()
        state = torch.randn(4, 272, device=device)

        output1 = network(state)
        output2 = network(state)

        assert torch.allclose(output1, output2)


class TestMaskedActionValues:
    """Test mask-based action selection (DEC-0034)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def network(self, device):
        return DQNNetwork(input_size=DEFAULT_INPUT_SIZE).to(device)

    def test_valid_actions_preserved(self, network, device):
        """Test that valid action Q-values are preserved."""
        state = torch.randn(4, 16, 17, device=device)
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)

        raw_q = network(state)
        masked_q = network.get_action_values(state, valid_mask)

        assert torch.allclose(raw_q, masked_q)

    def test_invalid_actions_masked(self, network, device):
        """Test that invalid action Q-values are set to -inf."""
        state = torch.randn(4, 16, 17, device=device)
        valid_mask = torch.zeros(4, 4, dtype=torch.bool, device=device)
        valid_mask[:, 0] = True  # Only action 0 valid

        masked_q = network.get_action_values(state, valid_mask)

        assert torch.all(masked_q[:, 1:] == float('-inf'))
        assert torch.all(masked_q[:, 0] != float('-inf'))

    def test_argmax_selects_valid_action(self, network, device):
        """Test that argmax on masked Q-values selects valid action."""
        state = torch.randn(4, 16, 17, device=device)
        valid_mask = torch.zeros(4, 4, dtype=torch.bool, device=device)
        valid_mask[0, 2] = True  # Game 0: only action 2 valid
        valid_mask[1, 1] = True  # Game 1: only action 1 valid
        valid_mask[2, 3] = True  # Game 2: only action 3 valid
        valid_mask[3, 0] = True  # Game 3: only action 0 valid

        masked_q = network.get_action_values(state, valid_mask)
        actions = masked_q.argmax(dim=1)

        assert actions[0] == 2
        assert actions[1] == 1
        assert actions[2] == 3
        assert actions[3] == 0

    def test_partial_valid_mask(self, network, device):
        """Test with some actions valid and some invalid."""
        state = torch.randn(2, 16, 17, device=device)
        valid_mask = torch.tensor([
            [True, True, False, False],
            [False, False, True, True]
        ], device=device)

        masked_q = network.get_action_values(state, valid_mask)

        # Check invalid actions are -inf
        assert masked_q[0, 2] == float('-inf')
        assert masked_q[0, 3] == float('-inf')
        assert masked_q[1, 0] == float('-inf')
        assert masked_q[1, 1] == float('-inf')

        # Check valid actions are finite
        assert torch.isfinite(masked_q[0, 0])
        assert torch.isfinite(masked_q[0, 1])
        assert torch.isfinite(masked_q[1, 2])
        assert torch.isfinite(masked_q[1, 3])
