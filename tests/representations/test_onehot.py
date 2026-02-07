"""
Tests for the OneHotRepresentation.

OneHotRepresentation is a pass-through that flattens (N, 16, 17) -> (N, 272).
"""

import pytest
import torch

from representations.onehot import OneHotRepresentation


class TestOneHotRepresentation:
    """Tests for OneHotRepresentation."""

    @pytest.fixture
    def rep(self):
        """Create representation instance."""
        return OneHotRepresentation()

    @pytest.fixture
    def sample_state(self):
        """Create sample one-hot game state."""
        # (N, 16, 17) one-hot boolean
        state = torch.zeros(4, 16, 17, dtype=torch.bool)
        # Set some positions to have tiles
        state[:, :, 0] = True  # All empty initially
        # Set position 0 to have tile value 1 (2^1 = 2)
        state[:, 0, 0] = False
        state[:, 0, 1] = True
        return state

    def test_output_shape_method(self, rep):
        """output_shape() returns correct shape."""
        assert rep.output_shape() == (272,)

    def test_forward_output_shape(self, rep, sample_state):
        """forward() produces correct output shape."""
        output = rep(sample_state)
        assert output.shape == (4, 272)

    def test_forward_output_dtype(self, rep, sample_state):
        """forward() produces float tensor."""
        output = rep(sample_state)
        assert output.dtype == torch.float32

    def test_forward_preserves_values(self, rep):
        """forward() preserves the one-hot values after flattening."""
        state = torch.zeros(2, 16, 17, dtype=torch.bool)
        # Set specific pattern
        state[0, 5, 3] = True  # Game 0, position 5, value 3
        state[1, 10, 7] = True  # Game 1, position 10, value 7

        output = rep(state)

        # Check that the flattened values are correct
        # Position 5, value 3 -> index 5*17 + 3 = 88
        assert output[0, 88] == 1.0
        # Position 10, value 7 -> index 10*17 + 7 = 177
        assert output[1, 177] == 1.0

    def test_forward_with_float_input(self, rep):
        """forward() handles float input correctly."""
        state = torch.zeros(3, 16, 17, dtype=torch.float32)
        state[:, :, 0] = 1.0

        output = rep(state)
        assert output.shape == (3, 272)
        assert output.dtype == torch.float32

    def test_config_optional(self):
        """Config is optional for OneHotRepresentation."""
        rep1 = OneHotRepresentation()
        rep2 = OneHotRepresentation({})
        rep3 = OneHotRepresentation({"ignored": "value"})

        # All should work identically
        state = torch.zeros(1, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True

        assert torch.equal(rep1(state), rep2(state))
        assert torch.equal(rep2(state), rep3(state))

    def test_no_learnable_parameters(self, rep):
        """OneHotRepresentation has no learnable parameters."""
        params = list(rep.parameters())
        assert len(params) == 0

    def test_batch_size_one(self, rep):
        """Works with batch size 1."""
        state = torch.zeros(1, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True
        output = rep(state)
        assert output.shape == (1, 272)

    def test_large_batch(self, rep):
        """Works with large batch sizes."""
        state = torch.zeros(128, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True
        output = rep(state)
        assert output.shape == (128, 272)

    def test_gradient_flow(self, rep):
        """Gradients flow through the representation."""
        state = torch.zeros(4, 16, 17, dtype=torch.float32, requires_grad=True)

        output = rep(state)
        loss = output.sum()
        loss.backward()

        assert state.grad is not None
        assert state.grad.shape == state.shape

    def test_gpu_support(self, rep):
        """Works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rep = rep.cuda()
        state = torch.zeros(4, 16, 17, dtype=torch.bool, device="cuda")
        state[:, :, 0] = True

        output = rep(state)
        assert output.device.type == "cuda"
        assert output.shape == (4, 272)
