"""Tests for DQN replay buffer."""

import pytest
import torch

from algorithms.dqn.replay_buffer import ReplayBuffer


class TestReplayBufferBasics:
    """Test basic replay buffer operations."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def buffer(self, device):
        return ReplayBuffer(capacity=100, device=device)

    def test_empty_buffer(self, buffer):
        """Test empty buffer properties."""
        assert len(buffer) == 0
        assert not buffer.is_ready(1)

    def test_push_single_batch(self, buffer, device):
        """Test pushing a single batch of transitions."""
        batch_size = 4

        buffer.push(
            state=torch.zeros(batch_size, 16, 17, dtype=torch.bool, device=device),
            action=torch.zeros(batch_size, dtype=torch.long, device=device),
            reward=torch.zeros(batch_size, dtype=torch.float32, device=device),
            next_state=torch.zeros(batch_size, 16, 17, dtype=torch.bool, device=device),
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),
            valid_mask=torch.ones(batch_size, 4, dtype=torch.bool, device=device),
        )

        assert len(buffer) == 4

    def test_push_multiple_batches(self, buffer, device):
        """Test pushing multiple batches."""
        for _ in range(5):
            buffer.push(
                state=torch.zeros(8, 16, 17, dtype=torch.bool, device=device),
                action=torch.zeros(8, dtype=torch.long, device=device),
                reward=torch.zeros(8, dtype=torch.float32, device=device),
                next_state=torch.zeros(8, 16, 17, dtype=torch.bool, device=device),
                done=torch.zeros(8, dtype=torch.bool, device=device),
                valid_mask=torch.ones(8, 4, dtype=torch.bool, device=device),
            )

        assert len(buffer) == 40

    def test_buffer_capacity(self, device):
        """Test buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=20, device=device)

        # Push 50 transitions
        for _ in range(10):
            buffer.push(
                state=torch.zeros(5, 16, 17, dtype=torch.bool, device=device),
                action=torch.zeros(5, dtype=torch.long, device=device),
                reward=torch.zeros(5, dtype=torch.float32, device=device),
                next_state=torch.zeros(5, 16, 17, dtype=torch.bool, device=device),
                done=torch.zeros(5, dtype=torch.bool, device=device),
                valid_mask=torch.ones(5, 4, dtype=torch.bool, device=device),
            )

        assert len(buffer) == 20  # Capped at capacity


class TestReplayBufferSampling:
    """Test replay buffer sampling."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def filled_buffer(self, device):
        buffer = ReplayBuffer(capacity=100, device=device)

        # Fill with identifiable data
        for i in range(10):
            batch_size = 4
            buffer.push(
                state=torch.full((batch_size, 16, 17), i, dtype=torch.bool, device=device),
                action=torch.full((batch_size,), i % 4, dtype=torch.long, device=device),
                reward=torch.full((batch_size,), float(i), dtype=torch.float32, device=device),
                next_state=torch.full((batch_size, 16, 17), i + 1, dtype=torch.bool, device=device),
                done=torch.zeros(batch_size, dtype=torch.bool, device=device),
                valid_mask=torch.ones(batch_size, 4, dtype=torch.bool, device=device),
            )

        return buffer

    def test_sample_shapes(self, filled_buffer, device):
        """Test sampled tensors have correct shapes."""
        states, actions, rewards, next_states, dones, valid_masks = filled_buffer.sample(16)

        assert states.shape == (16, 16, 17)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 16, 17)
        assert dones.shape == (16,)
        assert valid_masks.shape == (16, 4)

    def test_sample_dtypes(self, filled_buffer, device):
        """Test sampled tensors have correct dtypes."""
        states, actions, rewards, next_states, dones, valid_masks = filled_buffer.sample(8)

        assert states.dtype == torch.bool
        assert actions.dtype == torch.long
        assert rewards.dtype == torch.float32
        assert next_states.dtype == torch.bool
        assert dones.dtype == torch.bool
        assert valid_masks.dtype == torch.bool

    def test_sample_on_device(self, filled_buffer, device):
        """Test sampled tensors are on correct device."""
        states, actions, rewards, next_states, dones, valid_masks = filled_buffer.sample(8)

        # Compare device types (not exact device with index)
        assert states.device.type == device.type
        assert actions.device.type == device.type
        assert rewards.device.type == device.type
        assert next_states.device.type == device.type
        assert dones.device.type == device.type
        assert valid_masks.device.type == device.type

    def test_sample_randomness(self, filled_buffer):
        """Test that sampling is random."""
        # Use rewards which have unique values (0-9) in the filled_buffer
        samples1 = filled_buffer.sample(8)
        samples2 = filled_buffer.sample(8)

        # With 40 items and batch of 8, extremely unlikely to get same reward samples twice
        # Use rewards (index 2) which have distinct values
        assert not torch.equal(samples1[2], samples2[2])

    def test_is_ready(self, device):
        """Test is_ready with different min_size values."""
        buffer = ReplayBuffer(capacity=100, device=device)

        assert not buffer.is_ready(10)

        for i in range(5):
            buffer.push(
                state=torch.zeros(2, 16, 17, dtype=torch.bool, device=device),
                action=torch.zeros(2, dtype=torch.long, device=device),
                reward=torch.zeros(2, dtype=torch.float32, device=device),
                next_state=torch.zeros(2, 16, 17, dtype=torch.bool, device=device),
                done=torch.zeros(2, dtype=torch.bool, device=device),
                valid_mask=torch.ones(2, 4, dtype=torch.bool, device=device),
            )

        assert buffer.is_ready(10)
        assert not buffer.is_ready(20)


class TestEpisodeBoundaries:
    """Test episode boundary handling (DEC-0003)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_done_flag_preserved(self, device):
        """Test that done flags are correctly stored and retrieved."""
        buffer = ReplayBuffer(capacity=100, device=device)

        # Add transitions with mixed done flags
        done_flags = torch.tensor([False, True, False, True], device=device)
        buffer.push(
            state=torch.zeros(4, 16, 17, dtype=torch.bool, device=device),
            action=torch.zeros(4, dtype=torch.long, device=device),
            reward=torch.zeros(4, dtype=torch.float32, device=device),
            next_state=torch.zeros(4, 16, 17, dtype=torch.bool, device=device),
            done=done_flags,
            valid_mask=torch.ones(4, 4, dtype=torch.bool, device=device),
        )

        # Verify stored done flags
        assert buffer.dones[0] == False
        assert buffer.dones[1] == True
        assert buffer.dones[2] == False
        assert buffer.dones[3] == True

    def test_terminal_transition_stored(self, device):
        """Test that terminal transitions store next_state (not reset state)."""
        buffer = ReplayBuffer(capacity=100, device=device)

        # Create distinct terminal next_state
        terminal_next_state = torch.ones(1, 16, 17, dtype=torch.bool, device=device)
        terminal_next_state[:, :, 0] = False  # Not empty

        buffer.push(
            state=torch.zeros(1, 16, 17, dtype=torch.bool, device=device),
            action=torch.zeros(1, dtype=torch.long, device=device),
            reward=torch.tensor([100.0], device=device),
            next_state=terminal_next_state,
            done=torch.tensor([True], device=device),
            valid_mask=torch.zeros(1, 4, dtype=torch.bool, device=device),  # No valid actions (terminal)
        )

        # Verify the terminal next_state is stored correctly
        assert torch.equal(buffer.next_states[0], terminal_next_state[0])
        assert buffer.dones[0] == True
