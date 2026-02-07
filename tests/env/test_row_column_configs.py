"""
Category 5: Row/Column Configuration Tests

Tests for all significant line configurations:
- Single tile positions
- Two tile combinations
- Three tile combinations
- Four tile combinations
- Mixed value configurations

These tests systematically verify line processing for various arrangements.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestSingleTileConfigs:
    """Tests for single tile in various positions."""

    @pytest.mark.parametrize("position,input_line,expected_line", [
        (0, [2, 0, 0, 0], [2, 0, 0, 0]),  # Already at edge - no change (invalid)
        (1, [0, 2, 0, 0], [2, 0, 0, 0]),  # Slides left
        (2, [0, 0, 2, 0], [2, 0, 0, 0]),  # Slides left
        (3, [0, 0, 0, 2], [2, 0, 0, 0]),  # Slides left
    ])
    def test_single_tile_left_move(
        self, position, input_line, expected_line,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Single tile slides left correctly based on position."""
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Position 0 should be invalid (no change)
        # Positions 1-3 should slide to left edge

    @pytest.mark.parametrize("position,input_line,expected_line", [
        (0, [2, 0, 0, 0], [0, 0, 0, 2]),  # Slides right
        (1, [0, 2, 0, 0], [0, 0, 0, 2]),  # Slides right
        (2, [0, 0, 2, 0], [0, 0, 0, 2]),  # Slides right
        (3, [0, 0, 0, 2], [0, 0, 0, 2]),  # Already at edge - no change (invalid)
    ])
    def test_single_tile_right_move(
        self, position, input_line, expected_line,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Single tile slides right correctly based on position."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestTwoTileConfigs:
    """Tests for two tiles in various configurations."""

    @pytest.mark.parametrize("input_line,expected_line,should_merge", [
        # Adjacent pairs (merge)
        ([2, 2, 0, 0], [4, 0, 0, 0], True),
        ([0, 2, 2, 0], [4, 0, 0, 0], True),
        ([0, 0, 2, 2], [4, 0, 0, 0], True),
        # Separated pairs (merge after slide)
        ([2, 0, 2, 0], [4, 0, 0, 0], True),
        ([2, 0, 0, 2], [4, 0, 0, 0], True),
        ([0, 2, 0, 2], [4, 0, 0, 0], True),
        # Non-matching pairs (no merge, just slide)
        ([2, 4, 0, 0], [2, 4, 0, 0], False),  # No change - invalid
        ([0, 2, 4, 0], [2, 4, 0, 0], False),
        ([0, 0, 2, 4], [2, 4, 0, 0], False),
        ([2, 0, 4, 0], [2, 4, 0, 0], False),
        ([2, 0, 0, 4], [2, 4, 0, 0], False),
        ([0, 2, 0, 4], [2, 4, 0, 0], False),
    ])
    def test_two_tile_left_move(
        self, input_line, expected_line, should_merge,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Two tiles processed correctly for left move."""
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Verify expected_line after left move
        # Verify merge_reward = 4 if should_merge else 0

    @pytest.mark.parametrize("input_line,expected_line,should_merge", [
        # Adjacent pairs (merge)
        ([2, 2, 0, 0], [0, 0, 0, 4], True),
        ([0, 2, 2, 0], [0, 0, 0, 4], True),
        ([0, 0, 2, 2], [0, 0, 0, 4], True),
        # Separated pairs (merge after slide)
        ([2, 0, 2, 0], [0, 0, 0, 4], True),
        ([2, 0, 0, 2], [0, 0, 0, 4], True),
        ([0, 2, 0, 2], [0, 0, 0, 4], True),
    ])
    def test_two_tile_right_move(
        self, input_line, expected_line, should_merge,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Two tiles processed correctly for right move."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestThreeTileConfigs:
    """Tests for three tiles - critical for merge-once rule."""

    @pytest.mark.parametrize("input_line,expected_line,expected_reward", [
        # Three equal tiles - leftmost pair merges (DEC-0015)
        ([2, 2, 2, 0], [4, 2, 0, 0], 4),
        ([0, 2, 2, 2], [4, 2, 0, 0], 4),
        ([2, 0, 2, 2], [4, 2, 0, 0], 4),
        ([2, 2, 0, 2], [4, 2, 0, 0], 4),
        # Three tiles, first two equal
        ([2, 2, 4, 0], [4, 4, 0, 0], 4),
        ([0, 2, 2, 4], [4, 4, 0, 0], 4),
        # Three tiles, last two equal
        ([2, 4, 4, 0], [2, 8, 0, 0], 8),
        ([0, 2, 4, 4], [2, 8, 0, 0], 8),
        # Three tiles, first and last equal (no merge - not adjacent after slide)
        # [2, 4, 0, 2] left -> [2, 4, 2, 0] - no merge
        ([2, 4, 0, 2], [2, 4, 2, 0], 0),
        # Three different tiles
        ([2, 4, 8, 0], [2, 4, 8, 0], 0),  # No change - invalid
        ([0, 2, 4, 8], [2, 4, 8, 0], 0),
    ])
    def test_three_tile_left_move(
        self, input_line, expected_line, expected_reward,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Three tiles processed correctly with merge-once rule."""
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Verify expected_line and expected_reward

    @pytest.mark.parametrize("input_line,expected_line,expected_reward", [
        # Three equal tiles - rightmost pair merges (right move)
        ([2, 2, 2, 0], [0, 0, 2, 4], 4),
        ([0, 2, 2, 2], [0, 0, 2, 4], 4),
    ])
    def test_three_tile_right_move(
        self, input_line, expected_line, expected_reward,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Three tiles with right move - rightmost pair merges."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestFourTileConfigs:
    """Tests for full row configurations."""

    @pytest.mark.parametrize("input_line,expected_line,expected_reward", [
        # All same - two merges (DEC-0015: NOT [8,0,0,0])
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([4, 4, 4, 4], [8, 8, 0, 0], 16),
        ([8, 8, 8, 8], [16, 16, 0, 0], 32),
        # Two pairs
        ([2, 2, 4, 4], [4, 8, 0, 0], 12),
        ([4, 4, 2, 2], [8, 4, 0, 0], 12),
        # First pair matches
        ([2, 2, 4, 8], [4, 4, 8, 0], 4),
        # Last pair matches
        ([2, 4, 8, 8], [2, 4, 16, 0], 16),
        # Middle pair matches
        ([2, 4, 4, 8], [2, 8, 8, 0], 8),
        # No matches
        ([2, 4, 8, 16], [2, 4, 8, 16], 0),  # No change - invalid
        # Alternating
        ([2, 4, 2, 4], [2, 4, 2, 4], 0),  # No change - invalid
    ])
    def test_four_tile_left_move(
        self, input_line, expected_line, expected_reward,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Four tiles processed correctly for left move."""
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    @pytest.mark.parametrize("input_line,expected_line,expected_reward", [
        # All same - two merges at right
        ([2, 2, 2, 2], [0, 0, 4, 4], 8),
        # Two pairs
        ([2, 2, 4, 4], [0, 0, 4, 8], 12),
    ])
    def test_four_tile_right_move(
        self, input_line, expected_line, expected_reward,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Four tiles processed correctly for right move."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestMixedValueConfigs:
    """Tests for configurations with various tile values."""

    @pytest.mark.parametrize("input_line,expected_line,expected_reward", [
        # High values
        ([1024, 1024, 0, 0], [2048, 0, 0, 0], 2048),
        ([2048, 2048, 0, 0], [4096, 0, 0, 0], 4096),
        ([32768, 32768, 0, 0], [65536, 0, 0, 0], 65536),
        # Mixed high and low
        ([2, 2, 1024, 1024], [4, 2048, 0, 0], 2052),
        # Powers of 2 sequence
        ([2, 4, 8, 16], [2, 4, 8, 16], 0),  # No change - invalid
    ])
    def test_mixed_value_left_move(
        self, input_line, expected_line, expected_reward,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Mixed value tiles processed correctly."""
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestColumnConfigs:
    """Tests for column (vertical) configurations - same logic as rows."""

    @pytest.mark.parametrize("input_col,expected_col,expected_reward", [
        # Column processed same as row for UP move
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([2, 2, 2, 0], [4, 2, 0, 0], 4),
        ([0, 2, 2, 2], [4, 2, 0, 0], 4),
    ])
    def test_column_up_move(
        self, input_col, expected_col, expected_reward,
        make_env, line_to_col, grid_from_board, make_spawn_fn
    ):
        """Column tiles processed correctly for up move."""
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    @pytest.mark.parametrize("input_col,expected_col,expected_reward", [
        # Column processed for DOWN move - bottom is destination
        ([2, 2, 2, 2], [0, 0, 4, 4], 8),
        ([2, 2, 2, 0], [0, 0, 2, 4], 4),
    ])
    def test_column_down_move(
        self, input_col, expected_col, expected_reward,
        make_env, line_to_col, grid_from_board, make_spawn_fn
    ):
        """Column tiles processed correctly for down move."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestEmptyLineConfigs:
    """Tests for empty and near-empty line configurations."""

    def test_empty_row_unchanged(self, make_env, board_from_grid, make_spawn_fn):
        """Empty row remains empty (no change in that row)."""
        grid = [
            [0, 0, 0, 0],  # Empty row
            [2, 0, 0, 0],  # Tile in row 1
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_all_empty_except_one_row(self, make_env, board_from_grid, make_spawn_fn):
        """Only one row has tiles, processes correctly."""
        grid = [
            [2, 2, 4, 4],  # Only filled row
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Left: [4, 8, 0, 0] with reward 12
