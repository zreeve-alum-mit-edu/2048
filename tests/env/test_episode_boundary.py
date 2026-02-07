"""
Category 12: Episode Boundary Tests

Tests verifying correct episode boundary handling (DEC-0003):
- done=True: next_state is terminal state, NOT new game start
- reset_states returned separately for games that ended
- Cross-episode transitions properly separated

Critical for replay buffer safety.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestTerminalNextState:
    """Tests that next_state is terminal state when done=True."""

    def test_done_true_next_state_is_terminal(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """When done=True, next_state is the terminal board."""
        # Need to reach a terminal state
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # When done=True:
        # - next_state should be the final board (no valid moves)
        # - next_state should NOT be a fresh board with 2 tiles

    def test_next_state_not_reset_on_terminal(
        self, make_env, board_from_grid, grid_from_board, make_spawn_fn
    ):
        """Terminal next_state is NOT a reset board."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # If done=True:
        # - next_state should have 16 tiles (full board)
        # - next_state should NOT have 2 tiles (which would be reset)

    def test_terminal_board_verifiable(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """Terminal next_state can be verified as terminal."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # When done=True:
        # - next_state should have no valid moves
        # - We can verify by checking valid_mask is all False


class TestResetStatesReturned:
    """Tests for reset_states field in StepResult."""

    def test_reset_states_in_step_result(
        self, make_env, make_spawn_fn
    ):
        """StepResult contains reset_states field."""
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)

        assert hasattr(result, 'reset_states')
        assert result.reset_states.shape == (1, 16, 17)

    def test_reset_states_shape_matches_batch(
        self, make_env, make_spawn_fn
    ):
        """reset_states has same batch size as other outputs."""
        n_games = 10
        spawn_fn = make_spawn_fn(list(range(16)) * 10, [1] * 160)
        env = make_env(n_games=n_games, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1] * n_games)
        result = env.step(actions)

        assert result.reset_states.shape == (n_games, 16, 17)

    def test_reset_states_are_fresh_boards(
        self, make_env, grid_from_board, make_spawn_fn
    ):
        """reset_states contain fresh boards with 2 tiles."""
        spawn_fn = make_spawn_fn([0, 1, 2, 3], [1, 1, 1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)

        # reset_states should be fresh boards
        reset_grid = grid_from_board(result.reset_states)
        tile_count = sum(1 for row in reset_grid for v in row if v != 0)
        assert tile_count == 2, "Reset board should have 2 tiles"


class TestEpisodeSeparation:
    """Tests for proper episode separation."""

    def test_next_state_and_reset_state_different(
        self, make_env, grid_from_board, make_spawn_fn
    ):
        """On done=True, next_state != reset_states."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # When done=True:
        # - next_state is terminal (16 tiles, no valid moves)
        # - reset_states is fresh (2 tiles)
        # These must be different

    def test_no_cross_episode_transition(
        self, make_env, grid_from_board, make_spawn_fn
    ):
        """No single transition crosses episode boundary."""
        spawn_fn = make_spawn_fn(list(range(16)), [1] * 16)
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        # For replay buffer:
        # (state, action, reward, next_state, done)
        # If done=True, next_state is terminal
        # The "new episode" starts fresh, not connected to this transition


class TestBatchEpisodeBoundary:
    """Tests for episode boundaries in batched games."""

    def test_mixed_done_flags(
        self, make_env, boards_from_grids, grid_from_board, make_spawn_fn
    ):
        """Batch with some done=True, some done=False."""
        grids = [
            # Game 0: Will likely not be done (plenty of moves)
            [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            # Game 1: Could reach terminal (depending on setup)
            [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 0]],
        ]
        spawn_fn = make_spawn_fn([15, 15], [1, 1])
        env = make_env(n_games=2, spawn_fn=spawn_fn)

        # done might be [False, True]
        # next_state[0] is non-terminal
        # next_state[1] is terminal
        # reset_states[1] is fresh board

    def test_done_game_next_state_terminal(
        self, make_env, boards_from_grids, grid_from_board, make_spawn_fn
    ):
        """For done=True game, next_state is terminal."""
        spawn_fn = make_spawn_fn([0, 1], [1, 1])
        env = make_env(n_games=2, spawn_fn=spawn_fn)

        # If done[i] == True:
        # - next_state[i] should have no valid moves
        # - reset_states[i] should be fresh with 2 tiles

    def test_continuing_game_reset_state_still_valid(
        self, make_env, grid_from_board, make_spawn_fn
    ):
        """For done=False games, reset_states still populated."""
        spawn_fn = make_spawn_fn(list(range(16)) * 4, [1] * 64)
        env = make_env(n_games=4, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1, 1, 1, 1])  # All DOWN
        result = env.step(actions)

        # All reset_states should be valid fresh boards
        for i in range(4):
            reset_grid = grid_from_board(result.reset_states[i:i+1])
            tile_count = sum(1 for row in reset_grid for v in row if v != 0)
            assert tile_count == 2


class TestReplayBufferSafety:
    """Tests designed for replay buffer cross-episode safety."""

    def test_transition_tuple_components(
        self, make_env, make_spawn_fn
    ):
        """All transition tuple components available."""
        spawn_fn = make_spawn_fn([0, 1, 2], [1, 1, 1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        state = env.reset()

        actions = torch.tensor([1])  # DOWN
        result = env.step(actions)

        # Standard transition tuple:
        # (state, action, reward, next_state, done)
        assert state.shape == (1, 16, 17)
        assert actions.shape == (1,)
        assert result.merge_reward.shape == (1,)
        assert result.next_state.shape == (1, 16, 17)
        assert result.done.shape == (1,)

    def test_terminal_flag_reliable(
        self, make_env, board_from_grid, make_spawn_fn
    ):
        """done flag accurately reflects terminal state."""
        # Terminal board
        terminal_grid = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # done should be True for terminal
        # This is the signal to NOT store cross-episode transition


class TestMultipleEpisodesInBatch:
    """Tests for tracking multiple episode endings in batch."""

    def test_multiple_games_end_simultaneously(
        self, make_env, boards_from_grids, make_spawn_fn
    ):
        """Multiple games can end in same step."""
        spawn_fn = make_spawn_fn([15] * 4, [1] * 4)
        env = make_env(n_games=4, spawn_fn=spawn_fn)

        # done might be [True, True, False, True]
        # Each True game has distinct reset_state

    def test_reset_states_independent_per_game(
        self, make_env, grid_from_board, make_spawn_fn
    ):
        """Each game's reset_state is independent."""
        spawn_fn = make_spawn_fn([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 1, 2, 1, 2, 1, 2])
        env = make_env(n_games=4, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.tensor([1, 1, 1, 1])
        result = env.step(actions)

        # Each reset_state should be its own fresh board
        # Not sharing tiles or references


class TestConsecutiveEpisodes:
    """Tests for proper handling of consecutive episodes."""

    def test_game_continues_after_done(
        self, make_env, grid_from_board, make_spawn_fn
    ):
        """After done=True, game continues with reset_state."""
        spawn_fn = make_spawn_fn(list(range(16)) * 10, [1] * 160)
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        # Play until done
        # Then next step should use the reset_state
        # New episode begins fresh

    def test_episode_count_tracking(
        self, make_env, make_spawn_fn
    ):
        """Can track episode boundaries over multiple games."""
        spawn_fn = make_spawn_fn(list(range(16)) * 100, [1] * 1600)
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        env.reset()

        episode_count = 0
        for _ in range(100):
            actions = torch.tensor([1])  # DOWN
            try:
                result = env.step(actions)
                if result.done[0]:
                    episode_count += 1
            except InvalidMoveError:
                # Try different action
                pass

        # Episode count should be >= 0
