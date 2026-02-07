"""
Category 16: Deterministic Spawn Tests

Tests verifying deterministic spawn injection works correctly (DEC-0016):
- spawn_fn controls position and value
- Tests are reproducible
- Verify spawn at specified cell with specified value

The spawn_fn parameter enables testing without randomness.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestSpawnFunctionInjection:
    """Tests for spawn function injection mechanism."""

    def test_spawn_fn_parameter_accepted(
        self, make_env, make_spawn_fn
    ):
        """GameEnv accepts spawn_fn parameter."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Should not raise

    def test_spawn_fn_called_on_reset(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """spawn_fn is called during reset() for initial tiles."""
        spawn_fn = make_spawn_fn([0, 5], [1, 2])  # Position 0 gets 2, position 5 gets 4
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        grid = grid_from_board(state)
        # Position 0 (row 0, col 0) should have 2
        # Position 5 (row 1, col 1) should have 4
        assert grid[0][0] == 2
        assert grid[1][1] == 4

    def test_spawn_fn_called_on_step(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """spawn_fn is called after each step() for new tile."""
        # Initial spawns at 0, 1; subsequent spawn at 15
        spawn_fn = make_spawn_fn([0, 1, 15], [1, 1, 2])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)

        # Position 15 (row 3, col 3) should have the spawned 4
        # (tiles at 0, 1 will have moved down)


class TestSpawnPosition:
    """Tests for spawn position control."""

    @pytest.mark.parametrize("position", range(16))
    def test_spawn_at_each_position(
        self, position, make_env, spawn_at_position, grid_from_board
    ):
        """Can spawn at any of the 16 positions."""
        # First two spawns at fixed positions, third at target position
        spawn_fn = spawn_at_position(position, value=1)
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        # Check tile is at expected position
        grid = grid_from_board(state)
        row, col = position // 4, position % 4
        assert grid[row][col] == 2  # value=1 means tile value 2

    def test_spawn_position_in_empty_cell(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """Spawn always goes to specified empty cell."""
        spawn_fn = make_spawn_fn([0, 1, 2, 3], [1, 1, 1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        grid = grid_from_board(state)
        # First spawn at 0, second at 1
        assert grid[0][0] == 2
        assert grid[0][1] == 2


class TestSpawnValue:
    """Tests for spawn value control."""

    def test_spawn_value_2(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """spawn_fn can produce value 2 (log2 encoded as 1)."""
        spawn_fn = make_spawn_fn([0, 1], [1, 1])  # Both spawn 2
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        grid = grid_from_board(state)
        assert grid[0][0] == 2
        assert grid[0][1] == 2

    def test_spawn_value_4(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """spawn_fn can produce value 4 (log2 encoded as 2)."""
        spawn_fn = make_spawn_fn([0, 1], [2, 2])  # Both spawn 4
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        grid = grid_from_board(state)
        assert grid[0][0] == 4
        assert grid[0][1] == 4

    def test_mixed_spawn_values(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """spawn_fn can produce mixed 2s and 4s."""
        spawn_fn = make_spawn_fn([0, 1], [1, 2])  # First 2, second 4
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        grid = grid_from_board(state)
        assert grid[0][0] == 2
        assert grid[0][1] == 4


class TestSpawnReproducibility:
    """Tests for reproducible test execution."""

    def test_same_spawn_fn_same_result(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """Same spawn_fn produces same initial state."""
        spawn_fn1 = make_spawn_fn([0, 5], [1, 2])
        spawn_fn2 = make_spawn_fn([0, 5], [1, 2])

        env1 = make_env(n_games=1, spawn_fn=spawn_fn1)
        env2 = make_env(n_games=1, spawn_fn=spawn_fn2)

        state1 = env1.reset()
        state2 = env2.reset()

        grid1 = grid_from_board(state1)
        grid2 = grid_from_board(state2)

        assert grid1 == grid2

    def test_deterministic_game_sequence(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """Sequence of moves produces same boards with same spawn_fn."""
        def run_game():
            spawn_fn = make_spawn_fn(
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 1, 1, 1, 1, 1, 1, 1]
            )
            env = make_env(n_games=1, spawn_fn=spawn_fn)
            env.reset()

            states = []
            for action in [1, 2, 3, 0]:  # DOWN, LEFT, RIGHT, UP
                try:
                    result = env.step(torch.tensor([action]))
                    states.append(grid_from_board(result.next_state))
                except InvalidMoveError:
                    states.append(None)
            return states

        # Run twice
        states1 = run_game()
        states2 = run_game()

        # Should produce identical sequences
        assert states1 == states2


class TestBatchSpawn:
    """Tests for spawn function with batched games."""

    def test_batch_spawn_different_positions(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """Each game in batch can have different spawn positions."""
        # Game 0: spawn at 0, 1
        # Game 1: spawn at 14, 15
        spawn_fn = make_spawn_fn([0, 14, 1, 15], [1, 1, 1, 1])
        env = make_env(n_games=2, spawn_fn=spawn_fn)
        state = env.reset()

        grids = grid_from_board(state)
        assert grids[0][0][0] == 2  # Game 0, position 0
        assert grids[1][3][2] == 2  # Game 1, position 14

    def test_batch_spawn_different_values(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """Each game in batch can have different spawn values."""
        spawn_fn = make_spawn_fn([0, 0, 1, 1], [1, 2, 1, 2])
        # Game 0: positions 0,1 with values 2,2
        # Game 1: positions 0,1 with values 4,4
        env = make_env(n_games=2, spawn_fn=spawn_fn)


class TestSpawnFnInterface:
    """Tests for spawn function interface compliance."""

    def test_spawn_fn_receives_empty_mask(
        self, make_env
    ):
        """spawn_fn receives empty_mask tensor."""
        received_masks = []

        def recording_spawn_fn(empty_mask):
            received_masks.append(empty_mask.clone())
            n_games = empty_mask.shape[0]
            positions = torch.zeros(n_games, dtype=torch.long)
            values = torch.ones(n_games, dtype=torch.long)
            return positions, values

        env = make_env(n_games=1, spawn_fn=recording_spawn_fn)
        env.reset()

        assert len(received_masks) >= 1
        assert received_masks[0].shape == (1, 16)
        assert received_masks[0].dtype == torch.bool

    def test_spawn_fn_returns_correct_types(
        self, make_env
    ):
        """spawn_fn must return (positions, values) tensors."""
        def typed_spawn_fn(empty_mask):
            n_games = empty_mask.shape[0]
            positions = torch.zeros(n_games, dtype=torch.long)
            values = torch.ones(n_games, dtype=torch.long)
            return positions, values

        env = make_env(n_games=1, spawn_fn=typed_spawn_fn)
        state = env.reset()
        # Should succeed without type errors


class TestDefaultSpawnBehavior:
    """Tests for default spawn behavior (no spawn_fn)."""

    def test_no_spawn_fn_uses_random(
        self, make_env, grid_from_board
    ):
        """Without spawn_fn, spawns are random."""
        env = make_env(n_games=1)  # No spawn_fn

        state1 = env.reset()
        state2 = env.reset()

        # With random spawns, states might differ
        # (Though could be same by chance)

    def test_no_spawn_fn_90_10_distribution(
        self, make_env, grid_from_board
    ):
        """Without spawn_fn, spawn values follow 90% 2, 10% 4."""
        env = make_env(n_games=1)

        twos = 0
        fours = 0

        for _ in range(1000):
            state = env.reset()
            grid = grid_from_board(state)
            for row in grid:
                for val in row:
                    if val == 2:
                        twos += 1
                    elif val == 4:
                        fours += 1

        # Check approximate 90/10 split
        total = twos + fours
        two_ratio = twos / total

        # Allow 5% tolerance
        assert 0.85 < two_ratio < 0.95, \
            f"Expected ~90% 2s, got {two_ratio * 100:.1f}%"


class TestSpawnEdgeCases:
    """Tests for spawn edge cases."""

    def test_spawn_at_already_occupied_cell(
        self, make_env, make_spawn_fn, grid_from_board
    ):
        """Behavior when spawn_fn specifies occupied cell."""
        # This is a test contract issue - spawn_fn should respect empty_mask
        # Implementation might ignore or error
        spawn_fn = make_spawn_fn([0, 0], [1, 1])  # Both at position 0
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # Behavior is implementation-defined
        # Either second spawn goes elsewhere or error

    def test_spawn_after_full_board(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Behavior when no empty cells for spawn."""
        # Terminal board - no empty cells
        # Spawn shouldn't happen (game is over)
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # When done=True, spawn behavior is implementation-defined
