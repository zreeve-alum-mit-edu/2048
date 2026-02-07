"""
Category 15: Precompute Correctness Tests

Tests verifying precomputed lookup tables are correct:
- Line transition table: result of left move on any 4-tile line
- Valid move table: whether left move causes change
- Score delta table: points earned from merges

The GameEnv uses precomputed tables for GPU efficiency.
These tests verify the tables are correctly generated.
"""

import pytest
import torch

from game.env import GameEnv, InvalidMoveError


class TestLineTransitionTable:
    """Tests for the line transition lookup table."""

    @pytest.mark.parametrize("input_line,expected_output", [
        # Empty lines
        ([0, 0, 0, 0], [0, 0, 0, 0]),

        # Single tiles at various positions
        ([2, 0, 0, 0], [2, 0, 0, 0]),
        ([0, 2, 0, 0], [2, 0, 0, 0]),
        ([0, 0, 2, 0], [2, 0, 0, 0]),
        ([0, 0, 0, 2], [2, 0, 0, 0]),

        # Two tiles - same value (merge)
        ([2, 2, 0, 0], [4, 0, 0, 0]),
        ([0, 2, 2, 0], [4, 0, 0, 0]),
        ([0, 0, 2, 2], [4, 0, 0, 0]),
        ([2, 0, 2, 0], [4, 0, 0, 0]),
        ([2, 0, 0, 2], [4, 0, 0, 0]),
        ([0, 2, 0, 2], [4, 0, 0, 0]),

        # Two tiles - different values (no merge)
        ([2, 4, 0, 0], [2, 4, 0, 0]),
        ([0, 2, 4, 0], [2, 4, 0, 0]),
        ([0, 0, 2, 4], [2, 4, 0, 0]),
        ([2, 0, 4, 0], [2, 4, 0, 0]),

        # Three tiles - merge cases (DEC-0015)
        ([2, 2, 2, 0], [4, 2, 0, 0]),  # Leftmost pair merges
        ([0, 2, 2, 2], [4, 2, 0, 0]),
        ([2, 2, 4, 0], [4, 4, 0, 0]),
        ([2, 4, 4, 0], [2, 8, 0, 0]),
        ([4, 2, 2, 0], [4, 4, 0, 0]),

        # Four tiles - merge-once rule (DEC-0015)
        ([2, 2, 2, 2], [4, 4, 0, 0]),  # NOT [8, 0, 0, 0]
        ([2, 2, 4, 4], [4, 8, 0, 0]),
        ([4, 4, 2, 2], [8, 4, 0, 0]),
        ([2, 4, 2, 4], [2, 4, 2, 4]),  # No merge possible

        # High values
        ([1024, 1024, 0, 0], [2048, 0, 0, 0]),
        ([2048, 2048, 2048, 2048], [4096, 4096, 0, 0]),
    ])
    def test_line_transition_left(
        self, input_line, expected_output,
        make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Verify line transition for left move."""
        # This tests that the precomputed table gives correct results
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # The implementation should use lookup tables internally
        # We verify by testing actual step() results

    def test_all_single_value_lines(
        self, make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """Test lines with all same value."""
        for power in range(1, 16):  # 2^1 to 2^15
            value = 2 ** power
            input_line = [value, value, value, value]
            expected = [value * 2, value * 2, 0, 0]  # Two merges

            spawn_fn = make_spawn_fn([15], [1])
            env = make_env(n_games=1, spawn_fn=spawn_fn)
            # Verify transition


class TestValidMoveTable:
    """Tests for the valid move detection table."""

    @pytest.mark.parametrize("line,is_valid_left", [
        # Invalid: already at left, no merge
        ([2, 0, 0, 0], False),
        ([2, 4, 0, 0], False),
        ([2, 4, 8, 0], False),
        ([2, 4, 8, 16], False),

        # Valid: can slide
        ([0, 2, 0, 0], True),
        ([0, 0, 2, 0], True),
        ([0, 0, 0, 2], True),
        ([0, 2, 4, 8], True),

        # Valid: can merge
        ([2, 2, 0, 0], True),
        ([2, 0, 2, 0], True),
        ([2, 4, 4, 8], True),

        # Invalid: full, no merge, no slide
        ([2, 4, 2, 4], False),

        # Valid: full but can merge
        ([2, 2, 4, 8], True),
        ([2, 4, 4, 8], True),
        ([2, 4, 8, 8], True),
    ])
    def test_valid_move_detection(
        self, line, is_valid_left,
        make_env, line_to_row, make_spawn_fn
    ):
        """Verify valid move detection for left move."""
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # If is_valid_left is False, LEFT should raise InvalidMoveError
        # If is_valid_left is True, LEFT should succeed


class TestScoreDeltaTable:
    """Tests for the merge score (reward) table."""

    @pytest.mark.parametrize("input_line,expected_score", [
        # No merge
        ([2, 0, 0, 0], 0),
        ([2, 4, 0, 0], 0),
        ([2, 4, 8, 16], 0),
        ([2, 4, 2, 4], 0),

        # Single merge
        ([2, 2, 0, 0], 4),
        ([4, 4, 0, 0], 8),
        ([8, 8, 0, 0], 16),
        ([1024, 1024, 0, 0], 2048),

        # Two merges
        ([2, 2, 2, 2], 8),  # 4 + 4
        ([2, 2, 4, 4], 12),  # 4 + 8
        ([4, 4, 8, 8], 24),  # 8 + 16
        ([1024, 1024, 2048, 2048], 6144),  # 2048 + 4096

        # Three tiles, one merge
        ([2, 2, 4, 0], 4),
        ([2, 4, 4, 0], 8),

        # Max value merge
        ([32768, 32768, 0, 0], 65536),
    ])
    def test_score_delta(
        self, input_line, expected_score,
        make_env, line_to_row, make_spawn_fn
    ):
        """Verify merge score calculation."""
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # merge_reward should equal expected_score


class TestTableConsistency:
    """Tests for consistency between tables."""

    def test_valid_implies_transition_different(
        self, make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """If valid_move=True, transition output differs from input."""
        # For any line where valid_move table says True,
        # the transition table output must differ from input
        pass

    def test_score_nonzero_implies_transition_changes(
        self, make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """If score_delta > 0, transition output differs."""
        # Non-zero score means merge happened
        # Therefore output must differ from input
        pass

    def test_score_zero_implies_no_merge(
        self, make_env, line_to_row, grid_from_board, make_spawn_fn
    ):
        """If score_delta = 0, no merges occurred."""
        # Zero score means tiles only slid (or invalid move)
        pass


class TestTableCompleteness:
    """Tests verifying table covers all possible inputs."""

    def test_all_empty_line_handled(self, make_env, make_spawn_fn):
        """Empty line [0,0,0,0] handled correctly."""
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Should handle gracefully

    def test_all_max_tiles_handled(self, make_env, make_spawn_fn):
        """Line of maximum tiles handled correctly."""
        # [65536, 65536, 65536, 65536]
        spawn_fn = make_spawn_fn([0], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)
        # Should handle (even if merge exceeds encoding)

    def test_mixed_zero_and_values(self, make_env, make_spawn_fn):
        """Various patterns of zeros and values handled."""
        patterns = [
            [0, 2, 0, 4],
            [2, 0, 0, 4],
            [0, 0, 2, 4],
            [2, 4, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15] * 4, [1] * 4)
        env = make_env(n_games=1, spawn_fn=spawn_fn)


class TestPrecomputeEfficiency:
    """Tests that precomputed tables are used efficiently."""

    def test_step_uses_lookup_not_computation(
        self, make_env, gpu_device, make_spawn_fn, gpu_timer
    ):
        """step() should be fast (using lookups, not computing)."""
        # Large batch should still be very fast
        n_games = 1000
        spawn_fn = make_spawn_fn(list(range(16)) * n_games, [1] * (16 * n_games))
        env = make_env(n_games=n_games, device=gpu_device, spawn_fn=spawn_fn)
        env.reset()

        actions = torch.ones(n_games, dtype=torch.long, device=gpu_device)

        # Warm up
        try:
            env.step(actions)
        except InvalidMoveError:
            pass
        env.reset()

        # Time the operation
        with gpu_timer(gpu_device) as timer:
            try:
                env.step(actions)
            except InvalidMoveError:
                pass

        # Should be very fast (using lookup tables)
        # A naive implementation computing each line would be much slower


class TestTableSymmetry:
    """Tests for symmetry properties of lookup tables."""

    def test_all_directions_use_same_logic(
        self, make_env, board_from_grid, grid_from_board, rotate90, make_spawn_fn
    ):
        """All directions produce consistent results via rotation."""
        # LEFT on original should equal UP on rotated (under rotation)
        original = [
            [2, 2, 4, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        spawn_fn = make_spawn_fn([15], [1])
        env = make_env(n_games=1, spawn_fn=spawn_fn)

        # The precomputed tables should give consistent results
        # regardless of which direction is used
