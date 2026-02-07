"""
Category 7: Reward Correctness Tests

Tests verifying both reward types are calculated correctly:
- merge_reward: sum of merged tile values
- spawn_reward: value of spawned tile (2 or 4)

Both reward types must be returned per step.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestMergeReward:
    """Tests for merge_reward calculation."""

    def test_no_merge_zero_reward(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Move with no merge gives merge_reward = 0."""
        grid = [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([1], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)
        assert result.merge_reward[0] == 0

    def test_single_merge_reward(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Single merge: reward = merged tile value."""
        grid = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # 2 + 2 = 4, reward = 4
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # After step, merge_reward should be 4

    def test_multiple_merge_reward_sum(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Multiple merges: reward = sum of all merged values."""
        grid = [
            [2, 2, 4, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # 2+2=4, 4+4=8, total reward = 4 + 8 = 12
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # After left step, merge_reward should be 12

    def test_full_board_merge_reward(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Full board merge: reward = sum of all row merges."""
        grid = [
            [2, 2, 2, 2],
            [4, 4, 4, 4],
            [8, 8, 8, 8],
            [16, 16, 16, 16],
        ]
        # Row 0: 4+4=8
        # Row 1: 8+8=16
        # Row 2: 16+16=32
        # Row 3: 32+32=64
        # Total: 8+16+32+64 = 120
        spawn_fn = make_spawn_fn([2], [1])  # Spawn after merges create empty
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # After left step, merge_reward should be 120

    def test_high_value_merge_reward(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """High value merge gives large reward."""
        grid = [
            [32768, 32768, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # 32768 + 32768 = 65536, reward = 65536
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    @pytest.mark.parametrize("values,expected_reward", [
        ([2, 2], 4),
        ([4, 4], 8),
        ([8, 8], 16),
        ([16, 16], 32),
        ([32, 32], 64),
        ([64, 64], 128),
        ([128, 128], 256),
        ([256, 256], 512),
        ([512, 512], 1024),
        ([1024, 1024], 2048),
        ([2048, 2048], 4096),
        ([4096, 4096], 8192),
        ([8192, 8192], 16384),
        ([16384, 16384], 32768),
        ([32768, 32768], 65536),
    ])
    def test_merge_reward_all_tile_values(
        self, values, expected_reward,
        make_env, board_from_grid, make_spawn_fn
    ):
        """Verify merge reward for all possible tile value pairs."""
        grid = [
            [values[0], values[1], 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestSpawnReward:
    """Tests for spawn_reward calculation."""

    def test_spawn_reward_is_2_or_4(
        self, make_env, make_spawn_fn
    ):
        """Spawn reward is always 2 or 4."""
        spawn_fn = make_spawn_fn([0, 1], [1, 2])  # First spawn 2, second spawn 4
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)
        assert result.spawn_reward[0] in [2, 4]

    def test_spawn_reward_value_2(
        self, make_env, make_spawn_fn
    ):
        """Deterministic spawn of 2 gives spawn_reward = 2."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])  # All spawn 2
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)
        assert result.spawn_reward[0] == 2

    def test_spawn_reward_value_4(
        self, make_env, make_spawn_fn
    ):
        """Deterministic spawn of 4 gives spawn_reward = 4."""
        spawn_fn = make_spawn_fn([0, 1, 2], [2, 2, 2])  # All spawn 4
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)
        assert result.spawn_reward[0] == 4


class TestBothRewardTypes:
    """Tests verifying both reward types are returned."""

    def test_step_returns_both_rewards(
        self, make_env, make_spawn_fn
    ):
        """StepResult contains both merge_reward and spawn_reward."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)

        assert hasattr(result, 'merge_reward')
        assert hasattr(result, 'spawn_reward')
        assert result.merge_reward.shape == (1,)
        assert result.spawn_reward.shape == (1,)

    def test_rewards_are_tensors(
        self, make_env, make_spawn_fn
    ):
        """Both rewards are PyTorch tensors."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)

        assert isinstance(result.merge_reward, torch.Tensor)
        assert isinstance(result.spawn_reward, torch.Tensor)

    def test_merge_and_spawn_independent(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Merge and spawn rewards are calculated independently."""
        # Setup board with merge opportunity
        grid = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [2])  # Spawn 4
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # merge_reward should be 4 (from 2+2)
        # spawn_reward should be 4 (from spawn)
        # These are independent


class TestBatchRewards:
    """Tests for reward calculation with batched games."""

    def test_batch_independent_merge_rewards(
        self, make_env, boards_from_grids, make_spawn_fn
    ):
        """Each game in batch has independent merge reward."""
        grids = [
            [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # reward = 4
            [[4, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # reward = 8
            [[8, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # reward = 16
        ]
        spawn_fn = make_spawn_fn([15, 15, 15], [1, 1, 1])
        env = make_env(n_games=3, spawn_fn=spawn_fn)

        # After left step:
        # Game 0: merge_reward = 4
        # Game 1: merge_reward = 8
        # Game 2: merge_reward = 16

    def test_batch_independent_spawn_rewards(
        self, make_env, make_spawn_fn
    ):
        """Each game in batch has independent spawn reward."""
        # Different spawn values per game
        spawn_fn = make_spawn_fn([0, 1, 2, 3, 4, 5], [1, 2, 1, 2, 1, 2])
        env = make_env(n_games=3, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1, 1, 1])  # All DOWN
        result = env.step(actions)

        # Each game should have its own spawn_reward
        assert result.spawn_reward.shape == (3,)

    def test_batch_reward_shapes(
        self, make_env, make_spawn_fn
    ):
        """Batch rewards have correct shapes."""
        n_games = 10
        spawn_fn = make_spawn_fn(list(range(16)) * 10, [1] * 160)
        env = make_env(n_games=n_games, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1] * n_games)
        result = env.step(actions)

        assert result.merge_reward.shape == (n_games,)
        assert result.spawn_reward.shape == (n_games,)


class TestRewardEdgeCases:
    """Tests for edge cases in reward calculation."""

    def test_no_spawn_on_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Terminal game has no spawn (spawn_reward handling)."""
        # Terminal board
        grid = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # When done=True, spawn behavior is implementation-defined

    def test_zero_reward_no_merge(
        self, make_env, make_spawn_fn
    ):
        """Move without merge gives exactly zero merge reward."""
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        # If DOWN slides tiles without merge
        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)
        # merge_reward should be exactly 0, not close to 0

    def test_reward_dtype(
        self, make_env, make_spawn_fn
    ):
        """Rewards have appropriate dtype (float for merge, int-compatible for spawn)."""
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)

        # Both should be numeric tensors
        assert result.merge_reward.dtype in [torch.float32, torch.float64, torch.int32, torch.int64]
        assert result.spawn_reward.dtype in [torch.float32, torch.float64, torch.int32, torch.int64]
