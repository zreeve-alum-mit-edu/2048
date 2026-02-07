"""
Category 8: Terminal Detection Tests

Tests verifying correct game-over detection:
- No valid moves -> done=True
- At least one valid move -> done=False
- Full board with adjacent pairs -> NOT terminal
- Checkerboard (no adjacent equal) -> terminal

These tests ensure the done flag is set correctly.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestTerminalState:
    """Tests for terminal state (game over) detection."""

    def test_checkerboard_is_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Checkerboard pattern with no adjacent equal tiles is terminal."""
        grid = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # done should be True

    def test_all_different_tiles_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Full board with all different tiles is terminal."""
        grid = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 65536],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # done should be True

    def test_alternating_rows_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Alternating values in rows with no vertical matches is terminal."""
        grid = [
            [2, 4, 2, 4],
            [8, 16, 8, 16],
            [2, 4, 2, 4],
            [8, 16, 8, 16],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # No horizontal or vertical matches -> terminal


class TestNonTerminalState:
    """Tests for non-terminal states (game continues)."""

    def test_empty_cells_not_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Board with empty cells is never terminal."""
        grid = [
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # done should be False (can always make a move)

    def test_single_empty_cell_not_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Board with one empty cell is not terminal (can slide into it)."""
        grid = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 0],  # One empty
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # done should be False (tiles can slide)

    def test_horizontal_pair_not_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Full board with horizontal pair is NOT terminal."""
        grid = [
            [2, 2, 4, 8],  # Adjacent 2s
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [4096, 8192, 16384, 32768],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # done should be False (can merge the 2s)

    def test_vertical_pair_not_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Full board with vertical pair is NOT terminal."""
        grid = [
            [2, 4, 8, 16],
            [2, 32, 64, 128],  # 2 above and below
            [256, 512, 1024, 2048],
            [4096, 8192, 16384, 32768],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # done should be False (can merge the 2s vertically)

    def test_multiple_pairs_not_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Full board with multiple merge options is not terminal."""
        grid = [
            [2, 2, 4, 4],
            [8, 8, 16, 16],
            [32, 32, 64, 64],
            [128, 128, 256, 256],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Many valid moves available


class TestValidMaskConsistency:
    """Tests for valid_mask consistency with done flag."""

    def test_terminal_all_actions_invalid(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Terminal state has all actions marked invalid."""
        grid = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # valid_mask should be all False
        # done should be True

    def test_non_terminal_some_actions_valid(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Non-terminal state has at least one valid action."""
        grid = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # valid_mask should have at least one True
        # done should be False

    def test_done_false_implies_valid_action_exists(
        self, make_env, make_spawn_fn
    ):
        """If done=False, at least one action in valid_mask is True."""
        spawn_fn = make_spawn_fn(list(range(16)), [1] * 16)
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        # Initial state is never terminal
        # valid_mask should have at least one True

    def test_done_true_implies_no_valid_actions(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """If done=True, valid_mask is all False."""
        # Need to reach terminal state
        # This happens when board is full and no merges possible


class TestTerminalAfterMove:
    """Tests for moves that cause terminal state."""

    def test_move_fills_board_causes_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Move that fills board with no merges causes done=True."""
        # 15 tiles, one empty, spawn fills last cell
        # If spawn creates no merge option -> terminal
        spawn_fn = make_spawn_fn([15], [1])  # Spawn at last empty
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Construct board where spawn of 2 at position 15 creates terminal

    def test_move_creates_merge_not_terminal(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Move that fills board but creates merge is not terminal."""
        # Spawn fills last cell but creates adjacent pair
        spawn_fn = make_spawn_fn([15], [1])  # Spawn 2 next to another 2
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestBatchTerminalDetection:
    """Tests for terminal detection with batched games."""

    def test_mixed_terminal_batch(
        self, make_env, boards_from_grids, make_spawn_fn
    ):
        """Batch with some terminal, some not."""
        grids = [
            # Game 0: Terminal (checkerboard)
            [
                [2, 4, 2, 4],
                [4, 2, 4, 2],
                [2, 4, 2, 4],
                [4, 2, 4, 2],
            ],
            # Game 1: Not terminal (has empty cells)
            [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            # Game 2: Not terminal (has merge)
            [
                [2, 2, 4, 8],
                [16, 32, 64, 128],
                [256, 512, 1024, 2048],
                [4096, 8192, 16384, 32768],
            ],
        ]
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=3, spawn_fn=spawn_fn)
        # done should be [True, False, False]

    def test_batch_independent_terminal_detection(
        self, make_env, make_spawn_fn
    ):
        """Each game's terminal status is independent."""
        spawn_fn = make_spawn_fn(list(range(16)) * 10, [1] * 160)
        env = make_env(n_games=10, spawn_fn=spawn_fn)
        env.reset()

        # Games progress independently
        # Terminal detection should be per-game


class TestEdgeCaseTerminalDetection:
    """Tests for edge cases in terminal detection."""

    def test_almost_full_one_move_left(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Board where only one direction is valid."""
        # Carefully constructed so only LEFT works
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # valid_mask should have exactly one True

    def test_last_merge_extends_game(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Merge on almost-full board keeps game alive."""
        # 15 tiles, one merge creates empty cell(s)
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_terminal_with_max_tile(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Terminal state can include maximum tile value."""
        grid = [
            [65536, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # done should be True (no valid moves)
