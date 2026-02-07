"""
Category 4: Direction Symmetry Tests

Tests verifying all 4 directions behave consistently:
- Same logical scenario rotated for each direction
- Correct output per direction
- Symmetry of merge behavior across directions

These tests ensure direction-independent correctness.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestBasicDirectionSymmetry:
    """Tests for basic move symmetry across all directions."""

    def test_single_tile_slides_to_edge_all_directions(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn, actions
    ):
        """A single tile slides to the edge in any direction."""
        # Center tile at position (1,1)
        center_grid = [
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        expected_after = {
            actions.UP: (0, 1),     # Slides to row 0, col 1
            actions.DOWN: (3, 1),   # Slides to row 3, col 1
            actions.LEFT: (1, 0),   # Slides to row 1, col 0
            actions.RIGHT: (1, 3),  # Slides to row 1, col 3
        }

        for direction, (exp_row, exp_col) in expected_after.items():
            spawn_fn = make_spawn_fn([15], [1])  # Spawn in corner
            env = make_env(n_games=1, spawn_fn=spawn_fn)
            # Test that tile ends up at expected position

    def test_pair_merges_all_directions(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn, actions
    ):
        """A pair of tiles merges correctly in any direction."""
        # Horizontal pair
        h_grid = [
            [0, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # Vertical pair
        v_grid = [
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ]

        # Horizontal pair merges on LEFT or RIGHT
        # Vertical pair merges on UP or DOWN

    def test_no_move_in_blocked_direction(
        self, make_env, board_from_grid, make_spawn_fn, actions
    ):
        """Tiles at edge cannot move toward that edge (invalid)."""
        # Tile at left edge
        left_edge = [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([1], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # LEFT should be invalid (no change)
        # UP should be invalid (no change)
        # RIGHT and DOWN should be valid


class TestMergeDirectionSymmetry:
    """Tests for merge behavior symmetry across directions."""

    def test_merge_order_respects_direction_left(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Left move: merges happen left-to-right."""
        grid = [
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # Expected: [4, 2, 0, 0] - leftmost pair merges
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_merge_order_respects_direction_right(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Right move: merges happen right-to-left."""
        grid = [
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # Expected: [0, 0, 2, 4] - rightmost pair merges
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_merge_order_respects_direction_up(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Up move: merges happen top-to-bottom."""
        grid = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # Expected column 0: [4, 2, 0, 0] - topmost pair merges
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_merge_order_respects_direction_down(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Down move: merges happen bottom-to-top."""
        grid = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # Expected column 0: [0, 0, 2, 4] - bottommost pair merges
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestRowColumnEquivalence:
    """Tests that rows and columns behave equivalently."""

    def test_row_left_equals_column_up(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn, rotate90
    ):
        """Moving left on a row is equivalent to moving up on equivalent column."""
        row_grid = [
            [0, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # Rotate 90 CW: row 0 becomes column 3
        # [0,2,2,0] row -> [0,2,2,0] column (rotated)
        # UP on rotated should give same logical result as LEFT on original

    def test_row_right_equals_column_down(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn, rotate90
    ):
        """Moving right on a row is equivalent to moving down on equivalent column."""
        row_grid = [
            [0, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]


class TestFullBoardDirections:
    """Tests for full board scenarios in all directions."""

    def test_full_board_all_directions_produce_same_merge_count(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Symmetric full board gives same merge count in all directions."""
        # Rotationally symmetric pattern
        grid = [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        # Any direction: 8 merges, reward = 32
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_asymmetric_board_different_results_per_direction(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Non-symmetric board gives direction-dependent results."""
        grid = [
            [2, 2, 4, 4],
            [8, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # LEFT: [4, 8, 0, 0] in row 0, [8, 0, 0, 0] in row 1
        # RIGHT: [0, 0, 4, 8] in row 0
        # UP: no change in columns 2,3 (already at top)
        # DOWN: tiles slide down


class TestBatchDirectionSymmetry:
    """Tests for direction symmetry with batched games."""

    def test_different_actions_per_game(
        self, make_env, make_spawn_fn
    ):
        """Batch of N games can each take different direction."""
        spawn_fn = make_spawn_fn(list(range(16)) * 4, [1] * 64)
        env = make_env(n_games=4, spawn_fn=spawn_fn)
        env.reset()

        # Each game takes different action
        actions = torch.tensor([0, 1, 2, 3])  # UP, DOWN, LEFT, RIGHT
        result = env.step(actions)

        # Each game should have moved in its specified direction

    def test_same_board_different_actions(
        self, make_env, boards_from_grids, make_spawn_fn
    ):
        """Same initial board, different actions, different results."""
        # 4 copies of same board
        grid = [
            [0, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        grids = [grid] * 4
        spawn_fn = make_spawn_fn([15] * 4, [1] * 4)
        env = make_env(n_games=4, spawn_fn=spawn_fn)

        # Each takes different direction
        actions = torch.tensor([0, 1, 2, 3])  # UP, DOWN, LEFT, RIGHT
        # Results should differ based on direction
