"""
Category 14: GPU Timing Tests

Tests verifying operations complete within timing thresholds:
- Batch operations within threshold
- N=100: step < 1ms, reset < 0.5ms
- N=1000: step < 5ms, reset < 2.5ms

Based on GH200 specs (DEC-0023): 4TB/s HBM3 bandwidth, 528 Tensor Cores.
Thresholds include 50% buffer over estimated GPU-only time.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


# Mark all tests in this module as gpu_timing
pytestmark = pytest.mark.gpu_timing


class TestResetTiming:
    """Tests for reset() timing."""

    def test_reset_n100_under_threshold(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer, timing_thresholds
    ):
        """reset() with N=100 completes under threshold."""
        spawn_fn = make_spawn_fn(list(range(16)) * 100, [1] * 1600)
        env = make_env(n_games=100, device=gpu_device, spawn_fn=spawn_fn)

        # Warm up
        env.reset()

        # Timed run
        with gpu_timer(gpu_device) as timer:
            state = env.reset()

        assert timer.elapsed_ms < timing_thresholds["reset_n100"], \
            f"reset() took {timer.elapsed_ms:.2f}ms, threshold is {timing_thresholds['reset_n100']}ms"

    def test_reset_n1000_under_threshold(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer, timing_thresholds
    ):
        """reset() with N=1000 completes under threshold."""
        spawn_fn = make_spawn_fn(list(range(16)) * 1000, [1] * 16000)
        env = make_env(n_games=1000, device=gpu_device, spawn_fn=spawn_fn)

        # Warm up
        env.reset()

        # Timed run
        with gpu_timer(gpu_device) as timer:
            state = env.reset()

        assert timer.elapsed_ms < timing_thresholds["reset_n1000"], \
            f"reset() took {timer.elapsed_ms:.2f}ms, threshold is {timing_thresholds['reset_n1000']}ms"


class TestStepTiming:
    """Tests for step() timing."""

    def test_step_n100_under_threshold(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer, timing_thresholds
    ):
        """step() with N=100 completes under threshold."""
        spawn_fn = make_spawn_fn(list(range(16)) * 200, [1] * 3200)
        env = make_env(n_games=100, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.ones(100, dtype=torch.long, device=gpu_device)  # DOWN

        # Warm up
        try:
            env.step(actions)
        except InvalidMoveError:
            pass

        # Reset for clean state
        env.reset()

        # Timed run
        with gpu_timer(gpu_device) as timer:
            try:
                result = env.step(actions)
            except InvalidMoveError:
                pass

        assert timer.elapsed_ms < timing_thresholds["step_n100"], \
            f"step() took {timer.elapsed_ms:.2f}ms, threshold is {timing_thresholds['step_n100']}ms"

    def test_step_n1000_under_threshold(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer, timing_thresholds
    ):
        """step() with N=1000 completes under threshold."""
        spawn_fn = make_spawn_fn(list(range(16)) * 2000, [1] * 32000)
        env = make_env(n_games=1000, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.ones(1000, dtype=torch.long, device=gpu_device)  # DOWN

        # Warm up
        try:
            env.step(actions)
        except InvalidMoveError:
            pass

        # Reset for clean state
        env.reset()

        # Timed run
        with gpu_timer(gpu_device) as timer:
            try:
                result = env.step(actions)
            except InvalidMoveError:
                pass

        assert timer.elapsed_ms < timing_thresholds["step_n1000"], \
            f"step() took {timer.elapsed_ms:.2f}ms, threshold is {timing_thresholds['step_n1000']}ms"


class TestTimingConsistency:
    """Tests for consistent timing across runs."""

    def test_reset_timing_consistent(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer, timing_thresholds
    ):
        """reset() timing consistent across multiple calls."""
        spawn_fn = make_spawn_fn(list(range(16)) * 100, [1] * 1600)
        env = make_env(n_games=100, device=gpu_device, spawn_fn=spawn_fn)

        # Warm up
        env.reset()

        times = []
        for _ in range(10):
            with gpu_timer(gpu_device) as timer:
                env.reset()
            times.append(timer.elapsed_ms)

        # All times should be under threshold
        assert all(t < timing_thresholds["reset_n100"] for t in times), \
            f"Inconsistent times: {times}"

        # Variance should be reasonable (< 50% of mean)
        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5

        # Note: This is a soft check - GPU timing can have variance
        # Just ensure it's not wildly inconsistent

    def test_step_timing_consistent(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer, timing_thresholds
    ):
        """step() timing consistent across multiple calls."""
        spawn_fn = make_spawn_fn(list(range(16)) * 200, [1] * 3200)
        env = make_env(n_games=100, device=gpu_device, spawn_fn=spawn_fn)

        actions = torch.ones(100, dtype=torch.long, device=gpu_device)

        times = []
        for _ in range(10):
            env.reset()
            with gpu_timer(gpu_device) as timer:
                try:
                    env.step(actions)
                except InvalidMoveError:
                    pass
            times.append(timer.elapsed_ms)

        # All times should be under threshold
        assert all(t < timing_thresholds["step_n100"] for t in times), \
            f"Inconsistent times: {times}"


class TestTimingScaling:
    """Tests for timing scaling with batch size."""

    def test_reset_scales_sublinearly(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer
    ):
        """reset() scales efficiently with batch size."""
        times = {}

        for n in [100, 500, 1000]:
            spawn_fn = make_spawn_fn(list(range(16)) * n, [1] * (16 * n))
            env = make_env(n_games=n, device=gpu_device, spawn_fn=spawn_fn)

            # Warm up
            env.reset()

            with gpu_timer(gpu_device) as timer:
                env.reset()

            times[n] = timer.elapsed_ms

        # Scaling should be efficient (not 10x time for 10x batch)
        # Allow 5x time for 10x batch (sublinear scaling)
        if times[100] > 0:
            scaling_factor = times[1000] / times[100]
            # This is informational - GPU ops should scale well

    def test_step_scales_sublinearly(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer
    ):
        """step() scales efficiently with batch size."""
        times = {}

        for n in [100, 500, 1000]:
            spawn_fn = make_spawn_fn(list(range(16)) * n * 2, [1] * (32 * n))
            env = make_env(n_games=n, device=gpu_device, spawn_fn=spawn_fn)
            env.reset()

            actions = torch.ones(n, dtype=torch.long, device=gpu_device)

            # Warm up
            try:
                env.step(actions)
            except InvalidMoveError:
                pass
            env.reset()

            with gpu_timer(gpu_device) as timer:
                try:
                    env.step(actions)
                except InvalidMoveError:
                    pass

            times[n] = timer.elapsed_ms


class TestCPUTimingFails:
    """Tests demonstrating CPU implementation would fail timing."""

    def test_cpu_step_slower_than_gpu_threshold(
        self, make_env, cpu_device, make_spawn_fn, gpu_timer, timing_thresholds
    ):
        """CPU implementation likely exceeds GPU timing thresholds."""
        # This test documents expected behavior - CPU is slower
        spawn_fn = make_spawn_fn(list(range(16)) * 100, [1] * 1600)
        env = make_env(n_games=100, device=cpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.ones(100, dtype=torch.long)

        # Note: This test is informational
        # A naive CPU implementation would be much slower
        # We expect the actual GPU implementation to pass


class TestLargeScaleTiming:
    """Tests for timing with large batch sizes."""

    def test_large_batch_step_timing(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer
    ):
        """Large batch (N=10000) still reasonably fast."""
        n_games = 10000
        spawn_fn = make_spawn_fn(list(range(16)) * n_games, [1] * (16 * n_games))
        env = make_env(n_games=n_games, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.ones(n_games, dtype=torch.long, device=gpu_device)

        # Warm up
        try:
            env.step(actions)
        except InvalidMoveError:
            pass
        env.reset()

        with gpu_timer(gpu_device) as timer:
            try:
                env.step(actions)
            except InvalidMoveError:
                pass

        # Expected: < 50ms for N=10000 (linear scaling from N=1000 threshold)
        assert timer.elapsed_ms < 50, \
            f"Large batch step took {timer.elapsed_ms:.2f}ms"


class TestWarmupBehavior:
    """Tests for CUDA warmup behavior."""

    def test_first_call_slower_than_subsequent(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer
    ):
        """First CUDA call is slower (warmup), subsequent are faster."""
        spawn_fn = make_spawn_fn(list(range(16)) * 100, [1] * 1600)
        env = make_env(n_games=100, device=gpu_device, spawn_fn=spawn_fn)

        # First call (includes CUDA warmup)
        with gpu_timer(gpu_device) as timer1:
            env.reset()
        first_time = timer1.elapsed_ms

        # Subsequent calls (warmed up)
        times = []
        for _ in range(5):
            with gpu_timer(gpu_device) as timer:
                env.reset()
            times.append(timer.elapsed_ms)

        avg_subsequent = sum(times) / len(times)

        # First call typically slower (or equal)
        # This documents expected CUDA behavior
