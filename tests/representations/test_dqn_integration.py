"""
Tests for DQN integration with all representation types.

Per Milestone 4: DQN verified working with each representation.
"""

import pytest
import torch

from representations import (
    OneHotRepresentation,
    EmbeddingRepresentation,
    CNNRepresentation,
)
from algorithms.dqn.model import DQNNetwork
from algorithms.dqn.agent import DQNAgent


class TestDQNRepresentationIntegration:
    """Test that DQN works with all representation types."""

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sample_state(self, device):
        """Create sample one-hot game state."""
        state = torch.zeros(8, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True  # All empty
        # Add some tiles
        state[:, 0, 0] = False
        state[:, 0, 2] = True  # Tile value 2
        state[:, 5, 0] = False
        state[:, 5, 4] = True  # Tile value 4
        return state

    @pytest.fixture
    def valid_mask(self, device):
        """Create sample valid action mask."""
        mask = torch.ones(8, 4, dtype=torch.bool, device=device)
        mask[:, 3] = False  # Action 3 invalid for all games
        return mask

    def test_onehot_with_dqn_network(self, device, sample_state, valid_mask):
        """DQN network works with OneHotRepresentation."""
        rep = OneHotRepresentation().to(device)

        # Create network with correct input size
        input_size = rep.output_shape()[0]  # 272
        network = DQNNetwork(input_size=input_size).to(device)

        # Forward pass
        rep_output = rep(sample_state)
        q_values = network(rep_output)

        assert q_values.shape == (8, 4)
        assert not torch.isnan(q_values).any()

    def test_embedding_with_dqn_network(self, device, sample_state, valid_mask):
        """DQN network works with EmbeddingRepresentation."""
        rep = EmbeddingRepresentation({"embed_dim": 32}).to(device)

        # Create network with correct input size
        input_size = rep.output_shape()[0]  # 512
        network = DQNNetwork(input_size=input_size).to(device)

        # Forward pass
        rep_output = rep(sample_state)
        q_values = network(rep_output)

        assert q_values.shape == (8, 4)
        assert not torch.isnan(q_values).any()

    def test_cnn_default_with_dqn_network(self, device, sample_state, valid_mask):
        """DQN network works with default CNNRepresentation."""
        rep = CNNRepresentation().to(device)

        # Create network with correct input size
        input_size = rep.output_shape()[0]  # 576
        network = DQNNetwork(input_size=input_size).to(device)

        # Forward pass
        rep_output = rep(sample_state)
        q_values = network(rep_output)

        assert q_values.shape == (8, 4)
        assert not torch.isnan(q_values).any()

    def test_cnn_inception_with_dqn_network(self, device, sample_state, valid_mask):
        """DQN network works with Inception-style CNNRepresentation."""
        config = {
            "kernels": [
                {"size": [2, 2], "out_channels": 32, "stride": [1, 1]},
                {"size": [4, 1], "out_channels": 16, "stride": [1, 1]},
                {"size": [1, 4], "out_channels": 16, "stride": [1, 1]},
            ],
            "combine": "concat"
        }
        rep = CNNRepresentation(config).to(device)

        # Create network with correct input size
        input_size = rep.output_shape()[0]
        network = DQNNetwork(input_size=input_size).to(device)

        # Forward pass
        rep_output = rep(sample_state)
        q_values = network(rep_output)

        assert q_values.shape == (8, 4)
        assert not torch.isnan(q_values).any()

    def test_gradient_flow_onehot(self, device, sample_state):
        """Gradients flow through OneHot + DQN network."""
        rep = OneHotRepresentation().to(device)
        network = DQNNetwork(input_size=rep.output_shape()[0]).to(device)

        state = sample_state.float()
        state.requires_grad = True

        rep_output = rep(state)
        q_values = network(rep_output)
        loss = q_values.sum()
        loss.backward()

        assert state.grad is not None
        # Network parameters should have gradients
        for param in network.parameters():
            assert param.grad is not None

    def test_gradient_flow_embedding(self, device, sample_state):
        """Gradients flow through Embedding + DQN network."""
        rep = EmbeddingRepresentation({"embed_dim": 32}).to(device)
        network = DQNNetwork(input_size=rep.output_shape()[0]).to(device)

        state = sample_state.float()

        rep_output = rep(state)
        q_values = network(rep_output)
        loss = q_values.sum()
        loss.backward()

        # Embedding should have gradients
        assert rep.embedding.weight.grad is not None
        # Network parameters should have gradients
        for param in network.parameters():
            assert param.grad is not None

    def test_gradient_flow_cnn(self, device, sample_state):
        """Gradients flow through CNN + DQN network."""
        rep = CNNRepresentation().to(device)
        network = DQNNetwork(input_size=rep.output_shape()[0]).to(device)

        state = sample_state.float()
        state.requires_grad = True

        rep_output = rep(state)
        q_values = network(rep_output)
        loss = q_values.sum()
        loss.backward()

        # CNN conv layers should have gradients
        for branch in rep.branches:
            assert branch.weight.grad is not None
        # Network parameters should have gradients
        for param in network.parameters():
            assert param.grad is not None

    def test_dqn_agent_compatible_with_onehot(self, device, sample_state, valid_mask):
        """DQNAgent can use OneHotRepresentation output."""
        rep = OneHotRepresentation().to(device)
        agent = DQNAgent(
            device=device,
            hidden_layers=[128, 128],
        )

        # Agent expects raw state, but we need to modify network input size
        # For now, just verify the representation output is compatible
        rep_output = rep(sample_state)

        # Verify output shape matches expected DQN input
        assert rep_output.shape == (8, 272)

    def test_representation_output_shapes_match_spec(self, device):
        """Verify all representations produce flat output (OQ-5)."""
        reps = [
            OneHotRepresentation(),
            EmbeddingRepresentation({"embed_dim": 32}),
            CNNRepresentation(),
        ]

        state = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        state[:, :, 0] = True

        for rep in reps:
            rep = rep.to(device)
            output = rep(state)

            # Output should be 2D: (batch, features)
            assert output.dim() == 2
            # Output shape should match output_shape()
            assert output.shape[1:] == rep.output_shape()

    def test_different_embed_dims(self, device, sample_state):
        """DQN works with different embedding dimensions."""
        for embed_dim in [8, 16, 32, 64]:
            rep = EmbeddingRepresentation({"embed_dim": embed_dim}).to(device)
            network = DQNNetwork(input_size=rep.output_shape()[0]).to(device)

            rep_output = rep(sample_state)
            q_values = network(rep_output)

            assert q_values.shape == (8, 4)
            assert not torch.isnan(q_values).any()

    def test_different_cnn_configs(self, device, sample_state):
        """DQN works with different CNN configurations."""
        configs = [
            # Single 2x2 (default)
            {"kernels": [{"size": [2, 2], "out_channels": 64, "stride": [1, 1]}]},
            # Full row/column kernels
            {"kernels": [
                {"size": [4, 1], "out_channels": 32, "stride": [1, 1]},
                {"size": [1, 4], "out_channels": 32, "stride": [1, 1]},
            ]},
            # Large 3x3
            {"kernels": [{"size": [3, 3], "out_channels": 128, "stride": [1, 1]}]},
            # Strided
            {"kernels": [{"size": [2, 2], "out_channels": 64, "stride": [2, 2]}]},
        ]

        for config in configs:
            rep = CNNRepresentation(config).to(device)
            network = DQNNetwork(input_size=rep.output_shape()[0]).to(device)

            rep_output = rep(sample_state)
            q_values = network(rep_output)

            assert q_values.shape == (8, 4)
            assert not torch.isnan(q_values).any()
