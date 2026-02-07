"""
Tests for the EmbeddingRepresentation.

Per OQ-2: Shared embedding weights across positions.
Per OQ-5: Output is always flat for MLP input.
"""

import pytest
import torch

from representations.embedding import EmbeddingRepresentation


class TestEmbeddingRepresentation:
    """Tests for EmbeddingRepresentation."""

    @pytest.fixture
    def rep(self):
        """Create representation with default embed_dim=32."""
        return EmbeddingRepresentation({"embed_dim": 32})

    @pytest.fixture
    def sample_state(self):
        """Create sample one-hot game state."""
        state = torch.zeros(4, 16, 17, dtype=torch.bool)
        # Set all positions to empty (value 0)
        state[:, :, 0] = True
        # Set some positions to have tiles
        state[:, 0, 0] = False
        state[:, 0, 2] = True  # Position 0 has tile value 2 (4)
        state[:, 5, 0] = False
        state[:, 5, 4] = True  # Position 5 has tile value 4 (16)
        return state

    def test_output_shape_method(self, rep):
        """output_shape() returns correct shape for embed_dim=32."""
        # 16 positions * 32 embed_dim = 512
        assert rep.output_shape() == (512,)

    def test_output_shape_varies_with_embed_dim(self):
        """output_shape() changes with different embed_dim values."""
        rep8 = EmbeddingRepresentation({"embed_dim": 8})
        rep64 = EmbeddingRepresentation({"embed_dim": 64})

        assert rep8.output_shape() == (16 * 8,)  # 128
        assert rep64.output_shape() == (16 * 64,)  # 1024

    def test_forward_output_shape(self, rep, sample_state):
        """forward() produces correct output shape."""
        output = rep(sample_state)
        assert output.shape == (4, 512)  # 16 * 32

    def test_forward_output_dtype(self, rep, sample_state):
        """forward() produces float tensor."""
        output = rep(sample_state)
        assert output.dtype == torch.float32

    def test_shared_embedding_weights(self, rep):
        """Same tile value at different positions uses same embedding (OQ-2)."""
        state = torch.zeros(1, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True  # All empty

        # Set position 0 and position 5 to same tile value (3)
        state[:, 0, 0] = False
        state[:, 0, 3] = True
        state[:, 5, 0] = False
        state[:, 5, 3] = True

        output = rep(state)  # (1, 512)

        # The embeddings for position 0 and 5 should be identical
        # Position 0 embedding: output[0, 0:32]
        # Position 5 embedding: output[0, 5*32:6*32] = output[0, 160:192]
        pos0_embed = output[0, 0:32]
        pos5_embed = output[0, 160:192]

        assert torch.allclose(pos0_embed, pos5_embed)

    def test_different_values_different_embeddings(self, rep):
        """Different tile values produce different embeddings."""
        state = torch.zeros(2, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True  # All empty

        # Game 0: position 0 has value 1
        state[0, 0, 0] = False
        state[0, 0, 1] = True
        # Game 1: position 0 has value 10
        state[1, 0, 0] = False
        state[1, 0, 10] = True

        output = rep(state)

        # The embeddings for position 0 should differ between games
        game0_pos0 = output[0, 0:32]
        game1_pos0 = output[1, 0:32]

        # They should NOT be equal (different tile values)
        assert not torch.allclose(game0_pos0, game1_pos0)

    def test_has_learnable_parameters(self, rep):
        """EmbeddingRepresentation has learnable embedding weights."""
        params = list(rep.parameters())
        assert len(params) == 1  # Just the embedding weight

        # Embedding weight shape: (17, embed_dim)
        assert params[0].shape == (17, 32)

    def test_gradient_flow(self, rep):
        """Gradients flow through the representation."""
        state = torch.zeros(4, 16, 17, dtype=torch.float32)
        state[:, :, 0] = 1.0

        output = rep(state)
        loss = output.sum()
        loss.backward()

        # Check embedding gradients exist
        assert rep.embedding.weight.grad is not None
        assert rep.embedding.weight.grad.shape == (17, 32)

    def test_batch_size_one(self, rep):
        """Works with batch size 1."""
        state = torch.zeros(1, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True
        output = rep(state)
        assert output.shape == (1, 512)

    def test_large_batch(self, rep):
        """Works with large batch sizes."""
        state = torch.zeros(128, 16, 17, dtype=torch.bool)
        state[:, :, 0] = True
        output = rep(state)
        assert output.shape == (128, 512)

    def test_gpu_support(self, rep):
        """Works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rep = rep.cuda()
        state = torch.zeros(4, 16, 17, dtype=torch.bool, device="cuda")
        state[:, :, 0] = True

        output = rep(state)
        assert output.device.type == "cuda"
        assert output.shape == (4, 512)

    def test_default_embed_dim(self):
        """Default embed_dim is used if not specified."""
        rep = EmbeddingRepresentation({})
        # Default is 32
        assert rep.embed_dim == 32
        assert rep.output_shape() == (512,)

    def test_output_is_flat(self, rep, sample_state):
        """Output is 2D flat tensor (OQ-5)."""
        output = rep(sample_state)
        assert output.dim() == 2  # (N, features)

    def test_all_tile_values_embedded(self, rep):
        """All 17 possible tile values can be embedded."""
        state = torch.zeros(17, 16, 17, dtype=torch.bool)

        # Each game has a different tile value at position 0
        for value in range(17):
            state[value, :, 0] = True  # All positions empty
            state[value, 0, 0] = False
            state[value, 0, value] = True  # Position 0 has value 'value'

        # Should not raise
        output = rep(state)
        assert output.shape == (17, 512)
