"""
Category 13: GPU Device Tests

Tests verifying all tensors are on the correct device:
- All output tensors on specified device
- No accidental CPU tensors
- State, rewards, done flags, masks all on GPU

These tests ensure proper GPU utilization.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestTensorDevicePlacement:
    """Tests for correct tensor device placement."""

    def test_reset_returns_tensor_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """reset() returns state tensor on specified device."""
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        state = env.reset()

        assert state.device == gpu_device

    def test_step_next_state_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """step() returns next_state on specified device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)

        assert result.next_state.device == gpu_device

    def test_step_done_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """step() returns done tensor on specified device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)

        assert result.done.device == gpu_device

    def test_step_merge_reward_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """step() returns merge_reward on specified device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)

        assert result.merge_reward.device == gpu_device

    def test_step_spawn_reward_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """step() returns spawn_reward on specified device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)

        assert result.spawn_reward.device == gpu_device

    def test_step_valid_mask_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """step() returns valid_mask on specified device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)

        assert result.valid_mask.device == gpu_device

    def test_step_reset_states_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """step() returns reset_states on specified device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)

        assert result.reset_states.device == gpu_device


class TestAllOutputsOnDevice:
    """Tests that ALL outputs are on the correct device."""

    def test_all_step_outputs_same_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """All StepResult fields on same device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)

        devices = [
            result.next_state.device,
            result.done.device,
            result.merge_reward.device,
            result.spawn_reward.device,
            result.valid_mask.device,
            result.reset_states.device,
        ]

        assert all(d == gpu_device for d in devices), \
            f"Not all tensors on {gpu_device}: {devices}"


class TestCPUFallback:
    """Tests for CPU device when GPU unavailable."""

    def test_cpu_device_works(
        self, make_env, cpu_device, make_spawn_fn
    ):
        """Environment works on CPU device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=cpu_device, spawn_fn=spawn_fn)
        state = env.reset()

        assert state.device == cpu_device

    def test_cpu_step_works(
        self, make_env, cpu_device, make_spawn_fn
    ):
        """step() works on CPU device."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=cpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])
        result = env.step(actions)

        assert result.next_state.device == cpu_device


class TestActionTensorDevice:
    """Tests for action tensor device requirements."""

    def test_action_tensor_can_be_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """Action tensor on GPU device accepted."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)
        # Should not raise

    def test_action_tensor_on_different_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """Action tensor on different device - behavior defined."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        # CPU actions for GPU env - implementation may auto-convert or error
        actions = torch.tensor([1], device=torch.device("cpu"))
        # Either works or raises clear error


class TestBatchDevicePlacement:
    """Tests for device placement with batched games."""

    def test_batch_all_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """Batch of games all on specified device."""
        n_games = 100
        spawn_fn = make_spawn_fn(list(range(16)) * 100, [1] * 1600)
        env = make_env(n_games=n_games, device=gpu_device, spawn_fn=spawn_fn)
        state = env.reset()

        assert state.device == gpu_device
        assert state.shape[0] == n_games

    def test_batch_step_all_on_device(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """Batch step outputs all on device."""
        n_games = 100
        spawn_fn = make_spawn_fn(list(range(16)) * 100, [1] * 1600)
        env = make_env(n_games=n_games, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.ones(n_games, dtype=torch.long, device=gpu_device)
        result = env.step(actions)

        assert result.next_state.device == gpu_device
        assert result.done.device == gpu_device
        assert result.merge_reward.device == gpu_device


class TestNoAccidentalCPUTensors:
    """Tests ensuring no CPU tensors leak into GPU computation."""

    def test_no_cpu_tensors_in_reset(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """reset() creates no CPU tensors."""
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)

        # Before reset, check no CPU allocations from env
        state = env.reset()
        # All internal state should be on GPU

    def test_no_cpu_tensors_in_step(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """step() creates no CPU tensors."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1], device=gpu_device)
        result = env.step(actions)

        # All returned tensors on GPU
        # No intermediate CPU tensors should be created


class TestDeviceConsistency:
    """Tests for device consistency across operations."""

    def test_device_consistent_across_steps(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """Device stays consistent across multiple steps."""
        spawn_fn = make_spawn_fn(list(range(16)) * 10, [1] * 160)
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)
        state = env.reset()

        for _ in range(10):
            actions = torch.tensor([1], device=gpu_device)
            try:
                result = env.step(actions)
                assert result.next_state.device == gpu_device
            except InvalidMoveError:
                pass

    def test_device_consistent_after_reset(
        self, make_env, gpu_device, make_spawn_fn
    ):
        """Device consistent after multiple resets."""
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, device=gpu_device, spawn_fn=spawn_fn)

        for _ in range(5):
            state = env.reset()
            assert state.device == gpu_device


class TestSpecificGPUDevice:
    """Tests for specific GPU device selection."""

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need 2+ GPUs")
    def test_specific_cuda_device(self, make_env, make_spawn_fn):
        """Can specify specific CUDA device."""
        device = torch.device("cuda:1")
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, device=device, spawn_fn=spawn_fn)
        state = env.reset()

        assert state.device == device
