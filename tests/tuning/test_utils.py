"""Tests for tuning utilities."""

import pytest
import torch

from tuning.utils import create_representation, get_representation_output_dim
from representations.base import Representation
from representations.onehot import OneHotRepresentation
from representations.embedding import EmbeddingRepresentation
from representations.cnn import CNNRepresentation


class TestCreateRepresentation:
    """Test representation factory function."""

    def test_onehot_creation(self):
        """Test OneHot representation creation."""
        repr_module = create_representation("onehot", {})
        assert isinstance(repr_module, OneHotRepresentation)

    def test_embedding_creation(self):
        """Test Embedding representation creation."""
        repr_module = create_representation("embedding", {"embed_dim": 32})
        assert isinstance(repr_module, EmbeddingRepresentation)

    def test_cnn_2x2_creation(self):
        """Test CNN-2x2 representation creation."""
        repr_module = create_representation("cnn_2x2", {"cnn_channels": 64})
        assert isinstance(repr_module, CNNRepresentation)

    def test_cnn_4x1_creation(self):
        """Test CNN-4x1 representation creation."""
        repr_module = create_representation("cnn_4x1", {"cnn_channels": 64})
        assert isinstance(repr_module, CNNRepresentation)

    def test_cnn_multi_creation(self):
        """Test CNN-Multi representation creation."""
        repr_module = create_representation("cnn_multi", {"cnn_channels": 64})
        assert isinstance(repr_module, CNNRepresentation)

    def test_unknown_type_raises(self):
        """Test unknown representation type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown representation type"):
            create_representation("unknown_type", {})

    def test_all_representations_forward(self):
        """Test all representations can forward a batch."""
        device = torch.device("cpu")
        batch = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        batch[:, :, 0] = True  # Empty boards

        params_map = {
            "onehot": {},
            "embedding": {"embed_dim": 32},
            "cnn_2x2": {"cnn_channels": 64},
            "cnn_4x1": {"cnn_channels": 64},
            "cnn_multi": {"cnn_channels": 64},
        }

        for repr_type, params in params_map.items():
            repr_module = create_representation(repr_type, params).to(device)
            output = repr_module(batch)
            assert output.dim() == 2
            assert output.size(0) == 4


class TestGetRepresentationOutputDim:
    """Test output dimension calculator."""

    def test_onehot_dim(self):
        """Test OneHot output dimension is 272."""
        dim = get_representation_output_dim("onehot", {})
        assert dim == 272

    def test_embedding_dim(self):
        """Test Embedding output dimension is 16 * embed_dim."""
        dim = get_representation_output_dim("embedding", {"embed_dim": 32})
        assert dim == 16 * 32  # 512

    def test_cnn_2x2_dim(self):
        """Test CNN-2x2 output dimension."""
        dim = get_representation_output_dim("cnn_2x2", {"cnn_channels": 64})
        # 2x2 kernel on 4x4 -> 3x3 output, 64 channels = 64 * 9 = 576
        assert dim == 64 * 9
