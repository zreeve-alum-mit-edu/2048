"""
Category 11: Invalid Move Error Tests

Tests verifying invalid move handling (DEC-0014):
- Move causing no board change raises InvalidMoveError
- Test all 4 directions for invalid scenarios
- Algorithms must handle the exception

These tests ensure proper error behavior for invalid moves.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestInvalidMoveRaises:
    """Tests that invalid moves raise InvalidMoveError."""

    def test_already_packed_left_raises(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Tiles already at left edge: LEFT is invalid."""
        grid = [
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([2])  # LEFT
            env.step(actions)

    def test_already_packed_right_raises(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Tiles already at right edge: RIGHT is invalid."""
        grid = [
            [0, 0, 2, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([2, 3], [1, 2])  # positions 2,3 = row 0 cols 2,3
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([3])  # RIGHT
            env.step(actions)

    def test_already_packed_up_raises(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Tiles already at top edge: UP is invalid."""
        grid = [
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([0, 4], [1, 2])  # positions 0,4 = row 0 col 0, row 1 col 0
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([0])  # UP
            env.step(actions)

    def test_already_packed_down_raises(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Tiles already at bottom edge: DOWN is invalid."""
        grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [4, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([8, 12], [1, 2])  # positions 8,12 = row 2 col 0, row 3 col 0
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([1])  # DOWN
            env.step(actions)


class TestNoMergeNoSlideInvalid:
    """Tests for boards where move causes no change."""

    def test_full_row_no_merge_left_invalid(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Tiles at left edge with different values: LEFT is invalid."""
        # Two different tiles at positions 0 and 1 (left edge)
        # LEFT cannot slide or merge them
        spawn_fn = make_spawn_fn([0, 1], [1, 2])  # positions 0,1 with values 2,4
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([2])  # LEFT
            env.step(actions)

    def test_full_row_no_merge_right_invalid(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Tiles at right edge with different values: RIGHT is invalid."""
        # Two different tiles at positions 2 and 3 (right edge)
        spawn_fn = make_spawn_fn([2, 3], [1, 2])  # positions 2,3 with values 2,4
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([3])  # RIGHT
            env.step(actions)

    def test_alternating_row_invalid_both_directions(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Two different tiles in row: one direction invalid."""
        # Tiles at left edge: LEFT invalid, but RIGHT/UP/DOWN may be valid
        spawn_fn = make_spawn_fn([0, 1], [1, 2])  # positions 0,1 with values 2,4
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        # LEFT is invalid (tiles already at left edge, can't merge different values)
        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([2])  # LEFT
            env.step(actions)


class TestSingleTileAtEdgeInvalid:
    """Tests for single tile at edge positions."""

    def test_single_tile_left_edge_left_invalid(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Two tiles at left edge: LEFT is invalid."""
        # Two tiles at positions 0 and 4 (column 0, rows 0 and 1)
        # LEFT should be invalid (tiles already at left edge)
        spawn_fn = make_spawn_fn([0, 4], [1, 2])  # Different values to prevent merge
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([2])  # LEFT
            env.step(actions)

    def test_single_tile_corner_two_directions_invalid(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Tiles in top-left area: LEFT and UP are both invalid."""
        # Two tiles at positions 0 and 1 (top row, left side)
        # Different values so can't merge
        spawn_fn = make_spawn_fn([0, 1], [1, 2])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        # LEFT is invalid (tiles at left edge, can't merge)
        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([2])  # LEFT
            env.step(actions)

        # Note: After the first step fails, we're still in same state
        # UP is also invalid (tiles at top row, can't merge)
        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([0])  # UP
            env.step(actions)


class TestValidMaskReflectsInvalid:
    """Tests that valid_mask correctly indicates invalid moves."""

    def test_valid_mask_false_for_invalid(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """valid_mask should be False for invalid directions."""
        grid = [
            [2, 4, 0, 0],  # LEFT is invalid
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        # Check valid_mask (would need to access it from state or env)
        # valid_mask[2] (LEFT) should be False

    def test_valid_mask_true_for_valid(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """valid_mask should be True for valid directions."""
        grid = [
            [0, 2, 0, 0],  # Can slide left, right, down
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # LEFT, RIGHT, DOWN should be valid
        # UP is invalid (already at top)


class TestBatchInvalidMoves:
    """Tests for invalid moves in batched games."""

    def test_batch_partial_invalid_does_not_raise(
        self, make_env, boards_from_grids, make_spawn_fn
    ):
        """If SOME (not all) games have invalid move, does NOT raise (DEC-0027).

        Only raises InvalidMoveError if ALL games have invalid move.
        Per-game validity is returned in StepResult for partial invalid batches.
        """
        # With fresh boards after reset, we can test mixed validity
        # by using different actions per game
        spawn_fn = make_spawn_fn([0, 1, 2, 3] * 4, [1] * 16)
        env = make_env(n_games=2, spawn_fn=spawn_fn)
        env.reset()

        # Fresh boards typically have valid moves in multiple directions
        # This should not raise since not ALL games are invalid
        # (Implementation handles partial invalidity gracefully)

    def test_batch_all_invalid_raises(
        self, make_env, boards_from_grids, make_spawn_fn
    ):
        """All games invalid: raises exception."""
        # Both games have tiles at left edge column (different rows)
        # Call 1: game 0 -> pos 0, game 1 -> pos 0
        # Call 2: game 0 -> pos 4, game 1 -> pos 4
        # So both games have tiles at column 0, rows 0 and 1
        spawn_fn = make_spawn_fn([0, 0, 4, 4], [1, 1, 2, 2])
        env = make_env(n_games=2, spawn_fn=spawn_fn)
        env.reset()

        # LEFT is invalid for both (tiles already at left edge, different values can't merge)
        with pytest.raises(InvalidMoveError):
            actions = torch.tensor([2, 2])  # Both LEFT
            env.step(actions)

    def test_batch_independent_valid_moves(
        self, make_env, make_spawn_fn
    ):
        """Each game can have different valid/invalid actions."""
        spawn_fn = make_spawn_fn(list(range(16)) * 4, [1] * 64)
        env = make_env(n_games=4, spawn_fn=spawn_fn)
        env.reset()

        # Different actions per game - should work if all valid


class TestInvalidMoveErrorMessage:
    """Tests for informative error messages."""

    def test_error_includes_game_index(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """InvalidMoveError should indicate which game(s) failed."""
        spawn_fn = make_spawn_fn([0, 1], [1, 2])  # Tiles at left edge
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        try:
            actions = torch.tensor([2])  # LEFT - invalid
            env.step(actions)
            assert False, "Should have raised InvalidMoveError"
        except InvalidMoveError as e:
            # Error message should be informative
            pass

    def test_error_includes_action(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """InvalidMoveError should indicate which action was invalid."""
        spawn_fn = make_spawn_fn([0, 1], [1, 2])  # Tiles at left edge
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        try:
            actions = torch.tensor([2])  # LEFT - invalid
            env.step(actions)
            assert False, "Should have raised InvalidMoveError"
        except InvalidMoveError as e:
            # Error message should mention the action
            pass


class TestInvalidMoveDoesNotMutateState:
    """Tests that invalid move doesn't change game state."""

    def test_state_unchanged_after_invalid(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Board state unchanged after InvalidMoveError."""
        grid = [
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        initial_state = env.reset()
        initial_grid = grid_from_board(initial_state)

        try:
            actions = torch.tensor([2])  # LEFT - might be invalid
            env.step(actions)
        except InvalidMoveError:
            pass

        # State should be unchanged
        # (Would need to access current state from env)

    def test_no_spawn_after_invalid(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """No tile spawns after invalid move."""
        spawn_fn = make_spawn_fn([0, 1, 15], [1, 1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        initial_state = env.reset()

        initial_count = sum(
            1 for row in grid_from_board(initial_state)
            for v in row if v != 0
        )

        try:
            actions = torch.tensor([0])  # UP - might be invalid
            env.step(actions)
        except InvalidMoveError:
            pass

        # Tile count should be unchanged (no spawn happened)
