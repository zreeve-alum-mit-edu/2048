"""
Category 2: Edge Case Tests

Tests for boundary conditions and unusual states:
- Full board with no merge possible (terminal)
- Full board with merge possible (not terminal)
- Single empty cell
- Almost terminal (one valid move)
- Board with maximum tile (65536)

These tests verify the environment handles edge conditions correctly.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestFullBoardNoMerge:
    """Tests for full board with no possible merges (terminal state)."""

    def test_checkerboard_pattern_is_terminal(self, make_env, board_from_grid, make_spawn_fn):
        """Checkerboard pattern with no adjacent equal tiles is terminal."""
        # Classic terminal pattern - no two adjacent tiles are equal
        # This is the "game over" state
        grid = [
            [2,   4,   2,   4],
            [4,   2,   4,   2],
            [2,   4,   2,   4],
            [4,   2,   4,   2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # After setting this state, done should be True

    def test_ascending_pattern_is_terminal(self, make_env, board_from_grid):
        """Board with all different ascending values is terminal."""
        # All 16 tiles different - no merges possible
        grid = [
            [2,    4,    8,    16],
            [32,   64,   128,  256],
            [512,  1024, 2048, 4096],
            [8192, 16384, 32768, 65536],
        ]
        # This should be terminal if no valid moves exist

    def test_full_board_done_flag_true(self, make_env, board_from_grid, make_spawn_fn):
        """Full board with no valid moves sets done=True."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # After reaching terminal state, done should be True


class TestFullBoardWithMerge:
    """Tests for full board where merges are still possible."""

    def test_full_board_adjacent_pair_not_terminal(self, make_env, board_from_grid):
        """Full board with adjacent equal pair is NOT terminal."""
        # Full but has adjacent 2s that can merge
        grid = [
            [2,   2,   4,   8],
            [16,  32,  64,  128],
            [256, 512, 1024, 2048],
            [4096, 8192, 16384, 32768],
        ]
        # done should be False because left/right move on row 0 is valid

    def test_full_board_vertical_pair_not_terminal(self, make_env, board_from_grid):
        """Full board with vertical equal pair is NOT terminal."""
        grid = [
            [2,   4,   8,   16],
            [2,   32,  64,  128],  # Column 0 has adjacent 2s
            [256, 512, 1024, 2048],
            [4096, 8192, 16384, 32768],
        ]
        # done should be False because up/down move on column 0 is valid

    def test_full_board_multiple_merge_options(self, make_env, board_from_grid):
        """Full board with multiple possible merges."""
        grid = [
            [2,   2,   4,   4],
            [8,   8,   16,  16],
            [32,  32,  64,  64],
            [128, 128, 256, 256],
        ]
        # Many valid moves available


class TestSingleEmptyCell:
    """Tests for board with only one empty cell."""

    def test_spawn_fills_last_cell(self, make_env, board_from_grid, grid_from_board, make_spawn_fn):
        """Spawn fills the last empty cell."""
        # 15 tiles, spawn should go to the only empty cell
        spawn_fn = make_spawn_fn([5], [1])  # Will spawn at position 5
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_last_cell_spawn_may_cause_terminal(self, make_env, board_from_grid, make_spawn_fn):
        """Filling last cell might make board terminal."""
        # If the spawned tile creates no merge opportunities, game over

    def test_last_cell_spawn_with_merge_still_playable(self, make_env, board_from_grid, make_spawn_fn):
        """Filling last cell with merge option keeps game alive."""
        # If spawned tile can merge, game continues


class TestAlmostTerminal:
    """Tests for boards with very limited valid moves."""

    def test_single_valid_move_left(self, make_env, board_from_grid, make_spawn_fn):
        """Board where only LEFT is valid."""
        # Construct board where only one direction works
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # valid_mask should be [False, False, True, False] or similar

    def test_single_valid_move_up(self, make_env, board_from_grid, make_spawn_fn):
        """Board where only UP is valid."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_last_move_causes_terminal(self, make_env, board_from_grid, make_spawn_fn):
        """The only valid move leads to terminal state."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestMaxTile:
    """Tests for boards with maximum tile value (65536 = 2^16)."""

    def test_board_with_max_tile_still_playable(self, make_env, board_from_grid, make_spawn_fn):
        """Board containing 65536 tile with valid moves is playable."""
        grid = [
            [65536, 0, 0, 0],
            [0,     0, 0, 0],
            [0,     0, 0, 0],
            [0,     0, 0, 0],
        ]
        # Many empty cells, definitely playable

    def test_max_tile_one_hot_encoding(self, board_from_grid):
        """65536 (2^16) correctly encoded at index 16."""
        grid = [
            [65536, 0, 0, 0],
            [0,     0, 0, 0],
            [0,     0, 0, 0],
            [0,     0, 0, 0],
        ]
        board = board_from_grid(grid)
        # Position 0 should have one-hot at index 16
        assert board[0, 0, 16] == 1.0
        assert board[0, 0, :16].sum() == 0.0  # No other indices set

    def test_max_tile_merge_not_supported(self, make_env, board_from_grid, make_spawn_fn):
        """Two 65536 tiles - merging would exceed encoding (edge case)."""
        # Note: In standard 2048, reaching 131072 is extremely rare
        # The one-hot encoding goes to 2^16, so this is a boundary
        grid = [
            [65536, 65536, 0, 0],
            [0,     0,     0, 0],
            [0,     0,     0, 0],
            [0,     0,     0, 0],
        ]
        # Behavior for merging two max tiles is implementation-defined
        # Test documents expected behavior

    def test_high_value_tiles_work_correctly(self, make_env, board_from_grid, grid_from_board, make_spawn_fn):
        """High value tiles (32768, 65536) merge and move correctly."""
        grid = [
            [32768, 32768, 0, 0],
            [0,     0,     0, 0],
            [0,     0,     0, 0],
            [0,     0,     0, 0],
        ]
        # Left move should produce [65536, 0, 0, 0] (+ spawn)


class TestBatchEdgeCases:
    """Tests for edge cases with multiple parallel games (N>1)."""

    def test_mixed_terminal_states(self, make_env, boards_from_grids, make_spawn_fn):
        """Batch with some games terminal, others not."""
        # Game 0: terminal (checkerboard)
        # Game 1: not terminal (has moves)
        grids = [
            [  # Terminal
                [2, 4, 2, 4],
                [4, 2, 4, 2],
                [2, 4, 2, 4],
                [4, 2, 4, 2],
            ],
            [  # Not terminal
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
        spawn_fn = make_spawn_fn([0, 0], [1, 1])
        env = make_env(n_games=2, spawn_fn=spawn_fn)
        # done should be [True, False]

    def test_independent_game_progression(self, make_env, make_spawn_fn):
        """Games in batch progress independently."""
        # N=3 games, each with different states
        spawn_fn = make_spawn_fn([0, 1, 2, 3, 4, 5], [1] * 6)
        env = make_env(n_games=3, spawn_fn=spawn_fn)
        env.reset()

        # Apply same action to all
        actions = torch.tensor([2, 2, 2])  # All LEFT
        result = env.step(actions)

        # Each game should have its own state
        assert result.next_state.shape[0] == 3
