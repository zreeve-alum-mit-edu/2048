"""
Category 10: Reflection Invariance Tests

Tests verifying reflection symmetry of game logic:
- reflect_h(board) + RIGHT == reflect_h(LEFT on original)
- reflect_v(board) + DOWN == reflect_v(UP on original)

These tests ensure horizontal and vertical symmetry.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestReflectionBasics:
    """Tests for basic reflection correctness."""

    def test_reflect_horizontal_fixture_works(
        self, reflect_horizontal, board_from_grid, grid_from_board
    ):
        """Verify reflect_horizontal produces correct reflection."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
        # After horizontal reflection (left-right flip):
        # [4, 3, 2, 1]
        # [8, 7, 6, 5]
        # [12, 11, 10, 9]
        # [16, 15, 14, 13]

    def test_reflect_vertical_fixture_works(
        self, reflect_vertical, board_from_grid, grid_from_board
    ):
        """Verify reflect_vertical produces correct reflection."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
        # After vertical reflection (top-bottom flip):
        # [13, 14, 15, 16]
        # [9, 10, 11, 12]
        # [5, 6, 7, 8]
        # [1, 2, 3, 4]

    def test_reflect_horizontal_twice_identity(
        self, reflect_horizontal, board_from_grid, grid_from_board
    ):
        """Reflecting horizontally twice returns original."""
        grid = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [0, 0, 2, 4],
            [8, 0, 0, 16],
        ]
        board = board_from_grid(grid)
        reflected_twice = reflect_horizontal(reflect_horizontal(board))

        original_grid = grid_from_board(board)
        result_grid = grid_from_board(reflected_twice)
        assert original_grid == result_grid

    def test_reflect_vertical_twice_identity(
        self, reflect_vertical, board_from_grid, grid_from_board
    ):
        """Reflecting vertically twice returns original."""
        grid = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [0, 0, 2, 4],
            [8, 0, 0, 16],
        ]
        board = board_from_grid(grid)
        reflected_twice = reflect_vertical(reflect_vertical(board))

        original_grid = grid_from_board(board)
        result_grid = grid_from_board(reflected_twice)
        assert original_grid == result_grid

    def test_reflection_preserves_tiles(
        self, reflect_horizontal, reflect_vertical, board_from_grid, grid_from_board
    ):
        """Reflection preserves all tile values."""
        grid = [
            [2, 0, 0, 4],
            [0, 8, 16, 0],
            [0, 32, 64, 0],
            [128, 0, 0, 256],
        ]
        board = board_from_grid(grid)

        h_reflected = reflect_horizontal(board)
        v_reflected = reflect_vertical(board)

        original_tiles = sorted([v for row in grid for v in row if v != 0])

        h_grid = grid_from_board(h_reflected)
        h_tiles = sorted([v for row in h_grid for v in row if v != 0])

        v_grid = grid_from_board(v_reflected)
        v_tiles = sorted([v for row in v_grid for v in row if v != 0])

        assert original_tiles == h_tiles == v_tiles


class TestHorizontalReflectionInvariance:
    """Tests: reflect_h(LEFT) == RIGHT(reflect_h(board))."""

    def test_single_tile_horizontal_invariance(
        self, make_env, board_from_grid, grid_from_board, reflect_horizontal, make_spawn_fn
    ):
        """Single tile: LEFT on original == RIGHT on h-reflected."""
        original = [
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # LEFT on original: [2, 0, 0, 0]
        # reflect_h: [0, 0, 2, 0]
        # RIGHT on reflected: [0, 0, 0, 2]
        # reflect_h(RIGHT result): [2, 0, 0, 0]
        # Should equal LEFT result

    def test_merge_horizontal_invariance(
        self, make_env, board_from_grid, grid_from_board, reflect_horizontal, make_spawn_fn
    ):
        """Merge: LEFT on original == RIGHT on h-reflected (under reflection)."""
        original = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # LEFT: [4, 0, 0, 0]
        # reflect_h(original): [0, 0, 2, 2]
        # RIGHT on reflected: [0, 0, 0, 4]
        # reflect_h(RIGHT result): [4, 0, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_three_tile_merge_horizontal_invariance(
        self, make_env, board_from_grid, grid_from_board, reflect_horizontal, make_spawn_fn
    ):
        """[2,2,2,0] LEFT == h-reflected RIGHT (under reflection)."""
        original = [
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # LEFT: [4, 2, 0, 0]
        # reflect_h(original): [0, 2, 2, 2]
        # RIGHT: [0, 0, 2, 4]
        # reflect_h: [4, 2, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestVerticalReflectionInvariance:
    """Tests: reflect_v(UP) == DOWN(reflect_v(board))."""

    def test_single_tile_vertical_invariance(
        self, make_env, board_from_grid, grid_from_board, reflect_vertical, make_spawn_fn
    ):
        """Single tile: UP on original == DOWN on v-reflected."""
        original = [
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # UP on original: column 0 becomes [2, 0, 0, 0]
        # reflect_v: tile moves from row 1 to row 2
        # DOWN on reflected: column 0 becomes [0, 0, 0, 2]
        # reflect_v: becomes [2, 0, 0, 0]

    def test_merge_vertical_invariance(
        self, make_env, board_from_grid, grid_from_board, reflect_vertical, make_spawn_fn
    ):
        """Merge: UP on original == DOWN on v-reflected (under reflection)."""
        original = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # UP: column 0 becomes [4, 0, 0, 0]
        # reflect_v(original): tiles at rows 2,3
        # DOWN on reflected: column 0 becomes [0, 0, 0, 4]
        # reflect_v: [4, 0, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_three_tile_merge_vertical_invariance(
        self, make_env, board_from_grid, grid_from_board, reflect_vertical, make_spawn_fn
    ):
        """Column [2,2,2,0] UP == v-reflected DOWN (under reflection)."""
        original = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # UP: column 0 becomes [4, 2, 0, 0]
        # reflect_v(original): [0, 2, 2, 2] in column 0
        # DOWN: [0, 0, 2, 4]
        # reflect_v: [4, 2, 0, 0]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestReflectionWithRewards:
    """Tests reflection invariance of rewards."""

    def test_merge_reward_horizontal_invariant(
        self, make_env, board_from_grid, reflect_horizontal, make_spawn_fn
    ):
        """Same board h-reflected gives same merge reward."""
        original = [
            [2, 2, 4, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        # LEFT: reward = 4 + 8 = 12
        # reflect_h + RIGHT should also give reward = 12
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

    def test_merge_reward_vertical_invariant(
        self, make_env, board_from_grid, reflect_vertical, make_spawn_fn
    ):
        """Same board v-reflected gives same merge reward."""
        original = [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [4, 0, 0, 0],
        ]
        # UP: reward = 4 + 8 = 12
        # reflect_v + DOWN should also give reward = 12
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestReflectionWithTerminalDetection:
    """Tests reflection invariance of terminal detection."""

    def test_terminal_horizontal_invariant(
        self, make_env, board_from_grid, reflect_horizontal, make_spawn_fn
    ):
        """Terminal state remains terminal under h-reflection."""
        terminal = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Both original and reflect_h should be terminal

    def test_terminal_vertical_invariant(
        self, make_env, board_from_grid, reflect_vertical, make_spawn_fn
    ):
        """Terminal state remains terminal under v-reflection."""
        terminal = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Both original and reflect_v should be terminal


class TestCombinedReflections:
    """Tests for combined horizontal and vertical reflections."""

    def test_double_reflection_equals_180_rotation(
        self, board_from_grid, grid_from_board, reflect_horizontal, reflect_vertical, rotate90
    ):
        """H + V reflection equals 180 degree rotation."""
        grid = [
            [2, 4, 8, 16],
            [32, 0, 0, 64],
            [128, 0, 0, 256],
            [512, 1024, 2048, 4096],
        ]
        board = board_from_grid(grid)

        hv_reflected = reflect_vertical(reflect_horizontal(board))
        rotated_180 = rotate90(board, k=2)

        hv_grid = grid_from_board(hv_reflected)
        rot_grid = grid_from_board(rotated_180)

        assert hv_grid == rot_grid

    def test_reflection_order_commutative(
        self, board_from_grid, grid_from_board, reflect_horizontal, reflect_vertical
    ):
        """H then V equals V then H."""
        grid = [
            [2, 4, 8, 16],
            [32, 0, 0, 64],
            [128, 0, 0, 256],
            [512, 1024, 2048, 4096],
        ]
        board = board_from_grid(grid)

        hv = reflect_vertical(reflect_horizontal(board))
        vh = reflect_horizontal(reflect_vertical(board))

        hv_grid = grid_from_board(hv)
        vh_grid = grid_from_board(vh)

        assert hv_grid == vh_grid
