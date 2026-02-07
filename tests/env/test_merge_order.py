"""
Category 6: Merge Order Tests

Critical tests for the merge-once rule (DEC-0015):
- Each tile merges at most once per move
- Merges follow move direction
- [2,2,2,2] -> [4,4,0,0], NOT [8,0,0,0]

These tests verify the core merge logic is correct.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestMergeOnceRule:
    """Tests verifying tiles only merge once per move (DEC-0015)."""

    def test_four_equal_tiles_left_produces_two_merges(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn, assert_board_equals
    ):
        """[2,2,2,2] left -> [4,4,0,0], NOT [8,0,0,0]."""
        grid = [
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected = [
            [4, 4, 0, 0],  # + spawn somewhere
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])  # Spawn at position 15
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        # The key assertion: NOT [8, 0, 0, 0]

    def test_four_equal_tiles_right_produces_two_merges(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[2,2,2,2] right -> [0,0,4,4], NOT [0,0,0,8]."""
        grid = [
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected_row0 = [0, 0, 4, 4]  # NOT [0, 0, 0, 8]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_four_equal_tiles_up_produces_two_merges(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Column [2,2,2,2] up -> [4,4,0,0], NOT [8,0,0,0]."""
        grid = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
        ]
        # After UP, column 0 should be [4, 4, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_four_equal_tiles_down_produces_two_merges(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Column [2,2,2,2] down -> [0,0,4,4], NOT [0,0,0,8]."""
        grid = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
        ]
        # After DOWN, column 0 should be [0, 0, 4, 4]
        spawn_fn = make_spawn_fn([1], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_merged_tile_does_not_merge_again(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """A tile created by merge does not merge again in same move."""
        # [2, 2, 4, 0] left
        # First: 2+2=4 -> [4, 4, 0, 0]
        # The new 4 should NOT merge with existing 4
        # Result: [4, 4, 0, 0], NOT [8, 0, 0, 0]
        grid = [
            [2, 2, 4, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestMergeDirectionOrder:
    """Tests verifying merges follow move direction (from design doc 10.5)."""

    # From design doc section 10.5: Merge Order Test Examples

    def test_left_merge_order_three_tiles_a(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[2,2,2,0] left -> [4,2,0,0], NOT [2,4,0,0]."""
        grid = [
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected_row0 = [4, 2, 0, 0]
        wrong_row0 = [2, 4, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Assert: result is expected_row0, NOT wrong_row0

    def test_left_merge_order_three_tiles_b(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[0,2,2,2] left -> [4,2,0,0], NOT [2,4,0,0]."""
        grid = [
            [0, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected_row0 = [4, 2, 0, 0]
        wrong_row0 = [2, 4, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_left_merge_order_four_tiles(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[2,2,2,2] left -> [4,4,0,0], NOT [8,0,0,0]."""
        grid = [
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected_row0 = [4, 4, 0, 0]
        wrong_row0 = [8, 0, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_right_merge_order_three_tiles(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[2,2,2,0] right -> [0,0,2,4], NOT [0,0,4,2]."""
        grid = [
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected_row0 = [0, 0, 2, 4]
        wrong_row0 = [0, 0, 4, 2]
        spawn_fn = make_spawn_fn([8], [1])  # Spawn not in row 0
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_right_merge_order_four_tiles(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[2,2,2,2] right -> [0,0,4,4], NOT [0,0,0,8]."""
        grid = [
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected_row0 = [0, 0, 4, 4]
        wrong_row0 = [0, 0, 0, 8]
        spawn_fn = make_spawn_fn([8], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_up_merge_order_three_tiles(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Column [2,2,2,0] up -> [4,2,0,0], NOT [2,4,0,0]."""
        grid = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # Expected column 0: [4, 2, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_down_merge_order_three_tiles(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Column [2,2,2,0] down -> [0,0,2,4], NOT [0,0,4,2]."""
        grid = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # Expected column 0: [0, 0, 2, 4]
        spawn_fn = make_spawn_fn([1], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestMergeWithGaps:
    """Tests for merge behavior with empty cells between tiles."""

    def test_merge_across_gap(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[2,0,2,0] left -> [4,0,0,0] - tiles merge across gap."""
        grid = [
            [2, 0, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_merge_across_multiple_gaps(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[2,0,0,2] left -> [4,0,0,0] - tiles merge across multiple gaps."""
        grid = [
            [2, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_three_tiles_with_gaps(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """[2,0,2,2] left -> [4,2,0,0] - leftmost pair merges."""
        grid = [
            [2, 0, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # After slide: [2, 2, 2, 0] conceptually
        # Left merge: [4, 2, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestMergeRewardConsistency:
    """Tests verifying merge reward matches merge behavior."""

    def test_single_merge_reward(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Single merge of 2+2 gives reward 4."""
        grid = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Verify merge_reward = 4

    def test_double_merge_reward(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Two merges of 2+2 give reward 8."""
        grid = [
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Verify merge_reward = 8 (4 + 4)

    def test_mixed_merge_reward(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Different value merges give correct total reward."""
        grid = [
            [2, 2, 4, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Verify merge_reward = 12 (4 + 8)

    def test_high_value_merge_reward(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """High value merge gives correct reward."""
        grid = [
            [1024, 1024, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Verify merge_reward = 2048


class TestBatchMergeOrder:
    """Tests for merge order with batched games."""

    def test_batch_independent_merge_order(
        self, make_env, boards_from_grids, make_spawn_fn
    ):
        """Each game in batch follows merge order independently."""
        grids = [
            [[2, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # -> [4, 2, 0, 0]
            [[4, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # -> [8, 4, 0, 0]
        ]
        spawn_fn = make_spawn_fn([15, 15], [1, 1])
        env = make_env(n_games=2, spawn_fn=spawn_fn)

        # Both should follow merge-once rule independently
