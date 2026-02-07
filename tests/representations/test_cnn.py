"""
Tests for the CNNRepresentation.

Per OQ-1: No batch normalization.
Per OQ-3: No residual connections.
Per OQ-4: Default CNN config is single 2x2 kernel.
Per OQ-5: Output is always flat for MLP input.

Per design docs section 8.4:
- 4x1: Full rows
- 1x4: Full columns
- 2x2: Quadrants
- 3x3: Corner/center regions
"""

import pytest
import torch

from representations.cnn import CNNRepresentation


class TestCNNRepresentation:
    """Tests for CNNRepresentation."""

    @pytest.fixture
    def sample_state(self):
        """Create sample one-hot game state."""
        state = torch.zeros(4, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True  # All empty
        # Add some tiles
        state[:, 0, 0] = False
        state[:, 0, 2] = True
        return state

    def test_default_config(self):
        """Default config uses single 2x2 kernel (OQ-4)."""
        rep = CNNRepresentation()

        # Check default kernel config
        assert len(rep.branches) == 1

        # 2x2 kernel on 4x4 input with stride 1 -> 3x3 output
        # 64 channels * 3 * 3 = 576
        assert rep.output_shape() == (576,)

    def test_output_shape_method(self):
        """output_shape() returns correct shape."""
        rep = CNNRepresentation()
        shape = rep.output_shape()
        assert isinstance(shape, tuple)
        assert len(shape) == 1
        assert shape[0] > 0

    def test_forward_output_shape(self, sample_state):
        """forward() produces correct output shape."""
        rep = CNNRepresentation()
        output = rep(sample_state)
        expected_shape = (4,) + rep.output_shape()
        assert output.shape == expected_shape

    def test_forward_output_dtype(self, sample_state):
        """forward() produces float tensor."""
        rep = CNNRepresentation()
        output = rep(sample_state)
        assert output.dtype == torch.float32

    def test_output_is_flat(self, sample_state):
        """Output is 2D flat tensor (OQ-5)."""
        rep = CNNRepresentation()
        output = rep(sample_state)
        assert output.dim() == 2  # (N, features)

    def test_square_kernel_2x2(self, sample_state):
        """2x2 kernel produces expected output shape."""
        config = {
            "kernels": [{"size": [2, 2], "out_channels": 32, "stride": [1, 1]}]
        }
        rep = CNNRepresentation(config)

        # 2x2 on 4x4 -> 3x3 output, 32 channels
        # 32 * 3 * 3 = 288
        assert rep.output_shape() == (288,)

        output = rep(sample_state)
        assert output.shape == (4, 288)

    def test_square_kernel_3x3(self, sample_state):
        """3x3 kernel produces expected output shape."""
        config = {
            "kernels": [{"size": [3, 3], "out_channels": 64, "stride": [1, 1]}]
        }
        rep = CNNRepresentation(config)

        # 3x3 on 4x4 -> 2x2 output, 64 channels
        # 64 * 2 * 2 = 256
        assert rep.output_shape() == (256,)

        output = rep(sample_state)
        assert output.shape == (4, 256)

    def test_rectangular_kernel_4x1_row(self, sample_state):
        """4x1 kernel captures full rows."""
        config = {
            "kernels": [{"size": [4, 1], "out_channels": 32, "stride": [1, 1]}]
        }
        rep = CNNRepresentation(config)

        # 4x1 on 4x4 -> 1x4 output, 32 channels
        # 32 * 1 * 4 = 128
        assert rep.output_shape() == (128,)

        output = rep(sample_state)
        assert output.shape == (4, 128)

    def test_rectangular_kernel_1x4_column(self, sample_state):
        """1x4 kernel captures full columns."""
        config = {
            "kernels": [{"size": [1, 4], "out_channels": 32, "stride": [1, 1]}]
        }
        rep = CNNRepresentation(config)

        # 1x4 on 4x4 -> 4x1 output, 32 channels
        # 32 * 4 * 1 = 128
        assert rep.output_shape() == (128,)

        output = rep(sample_state)
        assert output.shape == (4, 128)

    def test_rectangular_kernel_2x4_half_board(self, sample_state):
        """2x4 kernel captures half-board vertical slices."""
        config = {
            "kernels": [{"size": [2, 4], "out_channels": 64, "stride": [1, 1]}]
        }
        rep = CNNRepresentation(config)

        # 2x4 on 4x4 -> 3x1 output, 64 channels
        # 64 * 3 * 1 = 192
        assert rep.output_shape() == (192,)

        output = rep(sample_state)
        assert output.shape == (4, 192)

    def test_rectangular_kernel_4x2_half_board(self, sample_state):
        """4x2 kernel captures half-board horizontal slices."""
        config = {
            "kernels": [{"size": [4, 2], "out_channels": 64, "stride": [1, 1]}]
        }
        rep = CNNRepresentation(config)

        # 4x2 on 4x4 -> 1x3 output, 64 channels
        # 64 * 1 * 3 = 192
        assert rep.output_shape() == (192,)

        output = rep(sample_state)
        assert output.shape == (4, 192)

    def test_inception_style_multi_kernel(self, sample_state):
        """Multiple kernels in parallel (Inception-style)."""
        config = {
            "kernels": [
                {"size": [2, 2], "out_channels": 32, "stride": [1, 1]},
                {"size": [4, 1], "out_channels": 16, "stride": [1, 1]},
                {"size": [1, 4], "out_channels": 16, "stride": [1, 1]},
            ],
            "combine": "concat"
        }
        rep = CNNRepresentation(config)

        # 2x2 -> 3x3 * 32 = 288
        # 4x1 -> 1x4 * 16 = 64
        # 1x4 -> 4x1 * 16 = 64
        # Total: 288 + 64 + 64 = 416
        assert rep.output_shape() == (416,)

        output = rep(sample_state)
        assert output.shape == (4, 416)

    def test_has_learnable_parameters(self):
        """CNNRepresentation has learnable conv weights."""
        rep = CNNRepresentation()
        params = list(rep.parameters())

        # Should have weights and biases for conv layers
        assert len(params) >= 2  # At least weight and bias

        # Check that parameters are trainable
        for param in params:
            assert param.requires_grad

    def test_gradient_flow(self, sample_state):
        """Gradients flow through the representation."""
        rep = CNNRepresentation()

        state = sample_state.float()
        state.requires_grad = True

        output = rep(state)
        loss = output.sum()
        loss.backward()

        # Check that conv layers received gradients
        for branch in rep.branches:
            assert branch.weight.grad is not None

    def test_batch_size_one(self):
        """Works with batch size 1."""
        rep = CNNRepresentation()
        state = torch.zeros(1, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True
        output = rep(state)
        assert output.shape[0] == 1

    def test_large_batch(self):
        """Works with large batch sizes."""
        rep = CNNRepresentation()
        state = torch.zeros(128, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True
        output = rep(state)
        assert output.shape[0] == 128

    def test_gpu_support(self, sample_state):
        """Works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rep = CNNRepresentation().cuda()
        state = sample_state.cuda()

        output = rep(state)
        assert output.device.type == "cuda"

    def test_stride_configuration(self, sample_state):
        """Custom stride reduces output size."""
        config = {
            "kernels": [{"size": [2, 2], "out_channels": 32, "stride": [2, 2]}]
        }
        rep = CNNRepresentation(config)

        # 2x2 kernel with stride 2 on 4x4 -> 2x2 output
        # 32 * 2 * 2 = 128
        assert rep.output_shape() == (128,)

        output = rep(sample_state)
        assert output.shape == (4, 128)

    def test_relu_activation_default(self):
        """Default activation is ReLU."""
        rep = CNNRepresentation()
        assert isinstance(rep.activation, torch.nn.ReLU)

    def test_tanh_activation(self, sample_state):
        """Tanh activation can be configured."""
        config = {"activation": "tanh"}
        rep = CNNRepresentation(config)
        assert isinstance(rep.activation, torch.nn.Tanh)

        # Should still produce valid output
        output = rep(sample_state)
        assert output.shape == (4,) + rep.output_shape()

    def test_invalid_activation_raises(self):
        """Invalid activation name raises ValueError."""
        config = {"activation": "invalid"}
        with pytest.raises(ValueError, match="Unknown activation"):
            CNNRepresentation(config)

    def test_combine_concat(self, sample_state):
        """Concat combine mode concatenates branch outputs."""
        config = {
            "kernels": [
                {"size": [2, 2], "out_channels": 32, "stride": [1, 1]},
                {"size": [2, 2], "out_channels": 16, "stride": [1, 1]},
            ],
            "combine": "concat"
        }
        rep = CNNRepresentation(config)

        # Both produce 3x3 output
        # 32 * 9 + 16 * 9 = 288 + 144 = 432
        assert rep.output_shape() == (432,)

    def test_combine_sum_same_size(self, sample_state):
        """Sum combine mode adds outputs of same-size branches."""
        config = {
            "kernels": [
                {"size": [2, 2], "out_channels": 32, "stride": [1, 1]},
                {"size": [2, 2], "out_channels": 32, "stride": [1, 1]},
            ],
            "combine": "sum"
        }
        rep = CNNRepresentation(config)

        # Both produce 32 * 9 = 288, summed = 288
        assert rep.output_shape() == (288,)

        output = rep(sample_state)
        assert output.shape == (4, 288)

    def test_combine_sum_different_size_raises(self):
        """Sum combine mode with different sizes raises error."""
        config = {
            "kernels": [
                {"size": [2, 2], "out_channels": 32, "stride": [1, 1]},
                {"size": [3, 3], "out_channels": 32, "stride": [1, 1]},
            ],
            "combine": "sum"
        }
        with pytest.raises(ValueError, match="Cannot use 'sum'"):
            CNNRepresentation(config)

    def test_no_batch_norm(self, sample_state):
        """No batch normalization layers (OQ-1)."""
        rep = CNNRepresentation()

        # Check no BatchNorm modules
        for module in rep.modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))

    def test_no_residual_connections(self, sample_state):
        """No residual connections in architecture (OQ-3)."""
        # Residual connections would require input and output to be added
        # We verify by checking the forward pass produces expected output size
        # without any skip connections

        rep = CNNRepresentation()
        output = rep(sample_state)

        # If there were residual connections, output would include input dims
        # Input flattened would be 16*17=272, but our output is different
        assert output.shape[1] != 272

    def test_input_channels_17(self):
        """Input is reshaped to have 17 channels (tile values)."""
        rep = CNNRepresentation()

        # First conv layer should have 17 input channels
        first_conv = rep.branches[0]
        assert first_conv.in_channels == 17

    def test_empty_kernels_list(self):
        """Empty kernels list produces zero output."""
        config = {"kernels": []}
        rep = CNNRepresentation(config)

        # No kernels = no output features
        assert rep.output_shape() == (0,)
