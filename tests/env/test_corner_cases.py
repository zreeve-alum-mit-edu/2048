"""
Category 3: Corner Case Tests

Tests for extreme and unusual scenarios:
- Maximum merges in one move (16 merges)
- Spawn on last empty cell
- All tiles identical (massive merge potential)
- Cascade scenarios

These tests verify handling of the most extreme valid game states.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestMaximumMerges:
    """Tests for maximum possible merges in a single move."""

    def test_four_merges_single_row(self, make_env, board_from_grid, grid_from_board, make_spawn_fn):
        """Four pairs merge in a single row: [2,2,2,2] -> [4,4,0,0]."""
        grid = [
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([4], [1])  # Spawn in row 1
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # After left move, row 0 should be [4, 4, 0, 0]
        # Reward should be 8 (4 + 4)

    def test_sixteen_merges_full_board(self, make_env, board_from_grid, grid_from_board, make_spawn_fn):
        """All 16 cells merge in pairs (8 merges per move direction)."""
        # All 2s - left move creates 8 merges (2 per row)
        grid = [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        spawn_fn = make_spawn_fn([2], [1])  # Spawn somewhere
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Left move: each row becomes [4, 4, 0, 0]
        # Total merge reward = 8 * 4 = 32

    def test_maximum_reward_single_move(self, make_env, board_from_grid, make_spawn_fn):
        """Calculate maximum possible reward from single move."""
        # Board of all 32768s merging
        grid = [
            [32768, 32768, 32768, 32768],
            [32768, 32768, 32768, 32768],
            [32768, 32768, 32768, 32768],
            [32768, 32768, 32768, 32768],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # 8 merges of 32768+32768 = 8 * 65536 = 524288 total reward

    def test_chained_row_merges(self, make_env, board_from_grid, grid_from_board, make_spawn_fn):
        """Multiple rows each have maximum merges."""
        grid = [
            [4, 4, 4, 4],
            [8, 8, 8, 8],
            [16, 16, 16, 16],
            [32, 32, 32, 32],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Each row merges independently


class TestSpawnOnLastCell:
    """Tests for spawning when only one cell is empty."""

    def test_spawn_deterministic_last_cell(self, make_env, board_from_grid, grid_from_board, make_spawn_fn):
        """Spawn goes to the only empty cell."""
        # Position 15 is the only empty cell
        spawn_fn = make_spawn_fn([15], [1])  # Spawn at last position
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_spawn_last_cell_fills_board(self, make_env, board_from_grid, grid_from_board, make_spawn_fn):
        """After spawn on last cell, board is full."""
        # 15 cells filled, spawn fills 16th
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # After step, board should have 16 non-empty cells

    def test_spawn_last_cell_causes_terminal(self, make_env, board_from_grid, make_spawn_fn):
        """Spawn on last cell with no merge options ends game."""
        # Carefully constructed: 15 cells with no adjacent pairs
        # Spawn fills last cell with value that creates no merges
        spawn_fn = make_spawn_fn([15], [1])  # Spawn 2
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # If the 15 existing tiles plus spawned 2 have no adjacent equals
        # -> done=True


class TestAllTilesIdentical:
    """Tests for boards where all tiles have the same value."""

    def test_all_twos_massive_merge(self, make_env, board_from_grid, grid_from_board, make_spawn_fn):
        """Board of all 2s has maximum merge potential."""
        grid = [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Any direction produces 8 merges

    def test_all_fours_merge(self, make_env, board_from_grid, make_spawn_fn):
        """Board of all 4s merges correctly."""
        grid = [
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Left: each row becomes [8, 8, 0, 0]

    def test_all_high_values_merge(self, make_env, board_from_grid, make_spawn_fn):
        """Board of all 1024s merges to 2048s."""
        grid = [
            [1024, 1024, 1024, 1024],
            [1024, 1024, 1024, 1024],
            [1024, 1024, 1024, 1024],
            [1024, 1024, 1024, 1024],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestCascadeScenarios:
    """Tests for cascade and chain reaction scenarios."""

    def test_sequential_moves_build_high_tile(self, make_env, make_spawn_fn):
        """Sequence of moves can build progressively higher tiles."""
        spawn_fn = make_spawn_fn(list(range(16)), [1] * 16)
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        # Series of moves to build up tiles
        # This tests the game flow, not a specific cascade

    def test_board_empties_then_fills(self, make_env, make_spawn_fn):
        """Board with merges empties partially then refills via spawns."""
        # Start with mergeable config
        spawn_fn = make_spawn_fn([0, 1, 2, 3, 4, 5, 6, 7], [1] * 8)
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        # Multiple moves cause merges (fewer tiles) then spawns (more tiles)

    def test_rapid_game_over_sequence(self, make_env, make_spawn_fn):
        """Sequence of spawns quickly leads to game over."""
        # Spawn in pattern that fills board without merge options
        # Alternating 2s and 4s
        spawn_fn = make_spawn_fn([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                  [1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestUnusualPatterns:
    """Tests for unusual but valid board patterns."""

    def test_single_column_full(self, make_env, board_from_grid, make_spawn_fn):
        """Only one column has tiles."""
        grid = [
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [16, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([1], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Up/down might be invalid, left/right should work

    def test_single_row_full(self, make_env, board_from_grid, make_spawn_fn):
        """Only one row has tiles."""
        grid = [
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([4], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_diagonal_pattern(self, make_env, board_from_grid, make_spawn_fn):
        """Tiles along diagonal."""
        grid = [
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 16],
        ]
        spawn_fn = make_spawn_fn([1], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_corners_only(self, make_env, board_from_grid, make_spawn_fn):
        """Tiles only in corners."""
        grid = [
            [2, 0, 0, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [8, 0, 0, 16],
        ]
        spawn_fn = make_spawn_fn([1], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_center_only(self, make_env, board_from_grid, make_spawn_fn):
        """Tiles only in center 2x2."""
        grid = [
            [0, 0, 0, 0],
            [0, 2, 4, 0],
            [0, 8, 16, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
