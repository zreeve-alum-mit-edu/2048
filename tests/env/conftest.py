"""
Pytest fixtures for GameEnv tests.

This module provides:
- Board creation and conversion utilities
- Deterministic spawn function factories
- GameEnv factory fixture
- Board transformation utilities (rotate, reflect)

All fixtures support the test-first development approach (DEC-0013).
"""

from typing import Callable, List, Optional, Tuple

import pytest
import torch
from torch import Tensor

from game.env import GameEnv, StepResult, InvalidMoveError


# =============================================================================
# BOARD CREATION UTILITIES
# =============================================================================

@pytest.fixture
def board_from_grid():
    """Create a one-hot board tensor from a 4x4 grid of raw tile values.

    The grid uses raw tile values (0, 2, 4, 8, 16, ..., 65536).
    These are converted to log2 indices for one-hot encoding:
    - 0 -> index 0 (empty)
    - 2 -> index 1
    - 4 -> index 2
    - ...
    - 65536 (2^16) -> index 16

    Returns:
        Callable that converts grid to (1, 16, 17) one-hot tensor
    """
    def _board_from_grid(
        grid: List[List[int]],
        device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Convert 4x4 grid to one-hot board tensor.

        Args:
            grid: 4x4 list of raw tile values (0, 2, 4, 8, ...)
            device: Target device for tensor

        Returns:
            (1, 16, 17) one-hot encoded board tensor
        """
        assert len(grid) == 4 and all(len(row) == 4 for row in grid), \
            "Grid must be 4x4"

        board = torch.zeros(1, 16, 17, device=device)
        for row_idx, row in enumerate(grid):
            for col_idx, value in enumerate(row):
                pos = row_idx * 4 + col_idx
                if value == 0:
                    idx = 0
                else:
                    # log2 of value gives the one-hot index
                    idx = int(torch.log2(torch.tensor(float(value))).item())
                board[0, pos, idx] = 1.0
        return board

    return _board_from_grid


@pytest.fixture
def grid_from_board():
    """Convert one-hot board tensor back to 4x4 grid of raw values.

    Returns:
        Callable that converts (N, 16, 17) tensor to list of 4x4 grids
    """
    def _grid_from_board(board: Tensor) -> List[List[List[int]]]:
        """Convert one-hot board tensor to grids.

        Args:
            board: (N, 16, 17) one-hot encoded board tensor

        Returns:
            List of N 4x4 grids with raw tile values
        """
        n_games = board.shape[0]
        grids = []

        for game_idx in range(n_games):
            grid = []
            for row_idx in range(4):
                row = []
                for col_idx in range(4):
                    pos = row_idx * 4 + col_idx
                    idx = board[game_idx, pos].to(torch.int64).argmax().item()
                    if idx == 0:
                        value = 0
                    else:
                        value = 2 ** idx
                    row.append(value)
                grid.append(row)
            grids.append(grid)

        return grids if n_games > 1 else grids[0]

    return _grid_from_board


@pytest.fixture
def boards_from_grids(board_from_grid):
    """Create batch of boards from multiple grids.

    Returns:
        Callable that converts list of grids to (N, 16, 17) tensor
    """
    def _boards_from_grids(
        grids: List[List[List[int]]],
        device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Convert multiple grids to batched board tensor.

        Args:
            grids: List of 4x4 grids
            device: Target device

        Returns:
            (N, 16, 17) one-hot tensor
        """
        boards = [board_from_grid(grid, device) for grid in grids]
        return torch.cat(boards, dim=0)

    return _boards_from_grids


# =============================================================================
# LINE UTILITIES (for row/column testing)
# =============================================================================

@pytest.fixture
def line_to_row():
    """Convert a 4-element line to a single-row board.

    Returns:
        Callable that creates a board with one non-empty row
    """
    def _line_to_row(
        line: List[int],
        row_idx: int = 0,
        device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Create board with specified row containing the line.

        Args:
            line: 4-element list of raw tile values
            row_idx: Which row (0-3) to place the line
            device: Target device

        Returns:
            (1, 16, 17) board with line in specified row
        """
        assert len(line) == 4, "Line must have 4 elements"
        grid = [[0, 0, 0, 0] for _ in range(4)]
        grid[row_idx] = line[:]

        board = torch.zeros(1, 16, 17, device=device)
        for col_idx, value in enumerate(line):
            pos = row_idx * 4 + col_idx
            if value == 0:
                idx = 0
            else:
                idx = int(torch.log2(torch.tensor(float(value))).item())
            board[0, pos, idx] = 1.0

        # Fill other positions with empty (index 0)
        for r in range(4):
            if r != row_idx:
                for c in range(4):
                    pos = r * 4 + c
                    board[0, pos, 0] = 1.0

        return board

    return _line_to_row


@pytest.fixture
def line_to_col():
    """Convert a 4-element line to a single-column board.

    Returns:
        Callable that creates a board with one non-empty column
    """
    def _line_to_col(
        line: List[int],
        col_idx: int = 0,
        device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Create board with specified column containing the line.

        Args:
            line: 4-element list of raw tile values
            col_idx: Which column (0-3) to place the line
            device: Target device

        Returns:
            (1, 16, 17) board with line in specified column
        """
        assert len(line) == 4, "Line must have 4 elements"

        board = torch.zeros(1, 16, 17, device=device)
        for row_idx, value in enumerate(line):
            pos = row_idx * 4 + col_idx
            if value == 0:
                idx = 0
            else:
                idx = int(torch.log2(torch.tensor(float(value))).item())
            board[0, pos, idx] = 1.0

        # Fill other positions with empty (index 0)
        for r in range(4):
            for c in range(4):
                if c != col_idx:
                    pos = r * 4 + c
                    board[0, pos, 0] = 1.0

        return board

    return _line_to_col


# =============================================================================
# DETERMINISTIC SPAWN UTILITIES
# =============================================================================

@pytest.fixture
def make_spawn_fn():
    """Factory for creating deterministic spawn functions.

    Returns:
        Callable that creates spawn functions with predetermined outputs
    """
    def _make_spawn_fn(
        positions: List[int],
        values: List[int]
    ) -> Callable[[Tensor], Tuple[Tensor, Tensor]]:
        """Create deterministic spawn function.

        Args:
            positions: List of cell indices (0-15) for each spawn
            values: List of tile values (1=2, 2=4 in log2) for each spawn

        Returns:
            spawn_fn compatible with GameEnv constructor
        """
        call_count = [0]  # Mutable container for closure

        def spawn_fn(empty_mask: Tensor) -> Tuple[Tensor, Tensor]:
            """Deterministic spawn function.

            Args:
                empty_mask: (N,16) boolean mask of empty cells

            Returns:
                positions: (N,) cell indices
                values: (N,) log2-encoded tile values
            """
            n_games = empty_mask.shape[0]
            device = empty_mask.device

            pos_out = torch.zeros(n_games, dtype=torch.long, device=device)
            val_out = torch.zeros(n_games, dtype=torch.long, device=device)

            for i in range(n_games):
                idx = (call_count[0] + i) % len(positions)
                pos_out[i] = positions[idx]
                val_out[i] = values[idx]

            call_count[0] += n_games
            return pos_out, val_out

        return spawn_fn

    return _make_spawn_fn


@pytest.fixture
def spawn_at_position():
    """Create spawn function that always spawns at a specific position.

    Returns:
        Callable that creates position-fixed spawn function
    """
    def _spawn_at_position(
        position: int,
        value: int = 1  # log2 encoded: 1=2, 2=4
    ) -> Callable[[Tensor], Tuple[Tensor, Tensor]]:
        """Create spawn function for fixed position.

        Args:
            position: Cell index (0-15)
            value: Log2-encoded tile value (1=2, 2=4)

        Returns:
            spawn_fn that always spawns at given position
        """
        def spawn_fn(empty_mask: Tensor) -> Tuple[Tensor, Tensor]:
            n_games = empty_mask.shape[0]
            device = empty_mask.device
            positions = torch.full((n_games,), position, dtype=torch.long, device=device)
            values = torch.full((n_games,), value, dtype=torch.long, device=device)
            return positions, values

        return spawn_fn

    return _spawn_at_position


# =============================================================================
# GAMEENV FACTORY
# =============================================================================

@pytest.fixture
def make_env():
    """Factory for creating GameEnv instances.

    Returns:
        Callable that creates GameEnv with specified parameters
    """
    def _make_env(
        n_games: int = 1,
        device: Optional[torch.device] = None,
        spawn_fn: Optional[Callable] = None
    ) -> GameEnv:
        """Create GameEnv instance.

        Args:
            n_games: Number of parallel games
            device: Target device (defaults to CPU)
            spawn_fn: Optional deterministic spawn function

        Returns:
            GameEnv instance
        """
        if device is None:
            device = torch.device("cpu")
        return GameEnv(n_games=n_games, device=device, spawn_fn=spawn_fn)

    return _make_env


@pytest.fixture
def device():
    """Get appropriate device for tests.

    Returns CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cpu_device():
    """Get CPU device."""
    return torch.device("cpu")


@pytest.fixture
def gpu_device():
    """Get GPU device (skips test if unavailable)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Use explicit index to ensure device comparison works correctly
    return torch.device("cuda:0")


# =============================================================================
# BOARD TRANSFORMATION UTILITIES
# =============================================================================

@pytest.fixture
def rotate90():
    """Rotate board 90 degrees clockwise.

    Returns:
        Callable that rotates (N, 16, 17) board k times
    """
    def _rotate90(board: Tensor, k: int = 1) -> Tensor:
        """Rotate board 90 degrees clockwise k times.

        Args:
            board: (N, 16, 17) one-hot board
            k: Number of 90-degree rotations (1-3)

        Returns:
            Rotated (N, 16, 17) board
        """
        k = k % 4
        if k == 0:
            return board.clone()

        n_games = board.shape[0]
        result = board.clone()

        # Position mapping for 90-degree clockwise rotation
        # Old position -> New position
        # (row, col) -> (col, 3-row)
        rotation_map = []
        for old_row in range(4):
            for old_col in range(4):
                new_row = old_col
                new_col = 3 - old_row
                old_pos = old_row * 4 + old_col
                new_pos = new_row * 4 + new_col
                rotation_map.append((old_pos, new_pos))

        for _ in range(k):
            new_result = torch.zeros_like(result)
            for old_pos, new_pos in rotation_map:
                new_result[:, new_pos, :] = result[:, old_pos, :]
            result = new_result

        return result

    return _rotate90


@pytest.fixture
def rotate_action():
    """Get corresponding action after board rotation.

    Returns:
        Callable that maps action after k rotations
    """
    def _rotate_action(action: int, k: int = 1) -> int:
        """Map action after k 90-degree clockwise rotations.

        If board is rotated k times, the "same" logical move
        corresponds to a different action.

        Original: up=0, down=1, left=2, right=3
        After 1 rotation: up->left, left->down, down->right, right->up

        Args:
            action: Original action (0-3)
            k: Number of rotations

        Returns:
            Equivalent action after rotation
        """
        k = k % 4
        # Rotation cycle: up -> left -> down -> right -> up
        # [0, 2, 1, 3] means: up(0)->left(2), down(1)->right(3), left(2)->down(1), right(3)->up(0)
        action_cycle = [0, 2, 1, 3]  # up, left, down, right in rotation order
        idx = action_cycle.index(action)
        new_idx = (idx + k) % 4
        return action_cycle[new_idx]

    return _rotate_action


@pytest.fixture
def reflect_horizontal():
    """Reflect board horizontally (left-right flip).

    Returns:
        Callable that reflects (N, 16, 17) board
    """
    def _reflect_horizontal(board: Tensor) -> Tensor:
        """Reflect board horizontally.

        Args:
            board: (N, 16, 17) one-hot board

        Returns:
            Horizontally reflected (N, 16, 17) board
        """
        result = torch.zeros_like(board)

        # Swap columns: 0<->3, 1<->2
        for row in range(4):
            for col in range(4):
                old_pos = row * 4 + col
                new_col = 3 - col
                new_pos = row * 4 + new_col
                result[:, new_pos, :] = board[:, old_pos, :]

        return result

    return _reflect_horizontal


@pytest.fixture
def reflect_vertical():
    """Reflect board vertically (top-bottom flip).

    Returns:
        Callable that reflects (N, 16, 17) board
    """
    def _reflect_vertical(board: Tensor) -> Tensor:
        """Reflect board vertically.

        Args:
            board: (N, 16, 17) one-hot board

        Returns:
            Vertically reflected (N, 16, 17) board
        """
        result = torch.zeros_like(board)

        # Swap rows: 0<->3, 1<->2
        for row in range(4):
            for col in range(4):
                old_pos = row * 4 + col
                new_row = 3 - row
                new_pos = new_row * 4 + col
                result[:, new_pos, :] = board[:, old_pos, :]

        return result

    return _reflect_vertical


# =============================================================================
# EXPECTED OUTPUT UTILITIES
# =============================================================================

@pytest.fixture
def assert_board_equals(grid_from_board):
    """Assert that a board matches expected grid.

    Returns:
        Callable for board equality assertion
    """
    def _assert_board_equals(
        board: Tensor,
        expected_grid: List[List[int]],
        msg: str = ""
    ):
        """Assert board matches expected grid.

        Args:
            board: (1, 16, 17) or (N, 16, 17) board tensor
            expected_grid: Expected 4x4 grid (for first game if N>1)
            msg: Optional assertion message
        """
        actual_grid = grid_from_board(board)
        if board.shape[0] > 1:
            actual_grid = actual_grid[0]

        assert actual_grid == expected_grid, \
            f"{msg}\nExpected:\n{_format_grid(expected_grid)}\nActual:\n{_format_grid(actual_grid)}"

    def _format_grid(grid: List[List[int]]) -> str:
        """Format grid for display."""
        lines = []
        for row in grid:
            lines.append(" ".join(f"{v:5d}" for v in row))
        return "\n".join(lines)

    return _assert_board_equals


@pytest.fixture
def assert_line_result():
    """Assert that a line operation produces expected result.

    Returns:
        Callable for line result assertion
    """
    def _assert_line_result(
        actual_line: List[int],
        expected_line: List[int],
        msg: str = ""
    ):
        """Assert line matches expected.

        Args:
            actual_line: Actual 4-element result
            expected_line: Expected 4-element result
            msg: Optional message
        """
        assert actual_line == expected_line, \
            f"{msg}\nExpected: {expected_line}\nActual: {actual_line}"

    return _assert_line_result


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@pytest.fixture
def gpu_timer():
    """Timer for GPU operations using CUDA events.

    Returns:
        Context manager that times GPU operations in milliseconds
    """
    class GPUTimer:
        """Context manager for timing GPU operations."""

        def __init__(self, device: torch.device):
            self.device = device
            self.elapsed_ms = 0.0
            self._start_event = None
            self._end_event = None

        def __enter__(self):
            if self.device.type == "cuda":
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
                self._start_event.record()
            else:
                import time
                self._start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            if self.device.type == "cuda":
                self._end_event.record()
                torch.cuda.synchronize()
                self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
            else:
                import time
                self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000

    def _create_timer(device: torch.device):
        return GPUTimer(device)

    return _create_timer


# =============================================================================
# TIMING THRESHOLDS (DEC-0023)
# =============================================================================

# GPU timing thresholds for GH200 (estimated with 50% buffer)
# See DEC-0023 for rationale
GPU_TIMING_THRESHOLDS = {
    "step_n100": 1.0,      # ms for step() with N=100
    "step_n1000": 5.0,     # ms for step() with N=1000
    "reset_n100": 0.5,     # ms for reset() with N=100
    "reset_n1000": 2.5,    # ms for reset() with N=1000
}


@pytest.fixture
def timing_thresholds():
    """Get GPU timing thresholds.

    Returns:
        Dict of operation -> threshold_ms
    """
    return GPU_TIMING_THRESHOLDS.copy()


# =============================================================================
# ACTION CONSTANTS
# =============================================================================

@pytest.fixture
def actions():
    """Action constants for readability.

    Returns:
        SimpleNamespace with UP, DOWN, LEFT, RIGHT
    """
    from types import SimpleNamespace
    return SimpleNamespace(UP=0, DOWN=1, LEFT=2, RIGHT=3)
