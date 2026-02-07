"""
GPU-native 2048 Game Environment.

This module provides the GameEnv class for running N parallel 2048 games
on GPU using PyTorch tensors.

Key design decisions:
- DEC-0014: Invalid moves raise InvalidMoveError
- DEC-0015: Merge-once rule enforced
- DEC-0027: Batch invalid moves only raise if ALL games invalid
- DEC-0029: dtypes - game state boolean, score int32
- DEC-0030: Internal state stored as one-hot (N, 16, 17) boolean
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from game.moves import execute_moves_batched, compute_valid_mask


class InvalidMoveError(Exception):
    """Raised when an action causes no board change.

    Per DEC-0014: GameEnv MUST raise exception on invalid move.
    Per DEC-0027: Only raises if ALL games in batch have invalid move.
    Algorithms are responsible for handling (penalty, force valid, etc.).
    """
    pass


@dataclass
class StepResult:
    """Result of a step in the game environment.

    Attributes:
        next_state: (N, 16, 17) one-hot encoded board states
        done: (N,) boolean flags indicating game over
        merge_reward: (N,) sum of merged tile values
        spawn_reward: (N,) value of spawned tile (2 or 4)
        valid_mask: (N, 4) boolean mask of valid actions
        reset_states: (N, 16, 17) fresh boards for games where done=True
    """
    next_state: Tensor
    done: Tensor
    merge_reward: Tensor
    spawn_reward: Tensor
    valid_mask: Tensor
    reset_states: Tensor


# Type alias for spawn function
# Args: empty_mask (N, 16) boolean
# Returns: (positions (N,), values (N,)) where values are log2 encoded (1=2, 2=4)
SpawnFn = Callable[[Tensor], Tuple[Tensor, Tensor]]


def _default_spawn_fn(empty_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Default spawn function with standard 2048 probabilities.

    Spawns 2 (90%) or 4 (10%) at random empty cell.

    Args:
        empty_mask: (N, 16) boolean mask of empty cells

    Returns:
        (positions, values): Cell indices and log2-encoded values
    """
    n_games = empty_mask.shape[0]
    device = empty_mask.device

    # Find empty positions and choose one randomly
    positions = torch.zeros(n_games, dtype=torch.long, device=device)

    for i in range(n_games):
        empty_indices = empty_mask[i].nonzero(as_tuple=True)[0]
        if len(empty_indices) > 0:
            rand_idx = torch.randint(len(empty_indices), (1,), device=device)
            positions[i] = empty_indices[rand_idx]

    # 90% chance of 2 (log2=1), 10% chance of 4 (log2=2)
    rand_vals = torch.rand(n_games, device=device)
    values = torch.where(rand_vals < 0.9,
                         torch.ones(n_games, dtype=torch.long, device=device),
                         torch.full((n_games,), 2, dtype=torch.long, device=device))

    return positions, values


class GameEnv:
    """GPU-native 2048 game environment for N parallel games.

    This environment runs entirely on GPU using PyTorch tensors.
    No CPU processing occurs during game steps (DEC-0001, DEC-0019).

    Game state is represented as (N, 16, 17) one-hot tensors:
    - N: number of parallel games
    - 16: board positions (4x4 flattened)
    - 17: possible values (0=empty, 1-16 = 2^1 to 2^16)

    Actions are integers 0-3:
    - 0: up
    - 1: down
    - 2: left
    - 3: right

    Key invariants (from design docs):
    - Merge-once rule: [2,2,2,2] -> [4,4,0,0], NOT [8,0,0,0] (DEC-0015)
    - Invalid moves raise InvalidMoveError (DEC-0014)
    - Episode boundary: done=True means next_state is terminal (DEC-0003)
    - Tile spawn: 2 (90%) or 4 (10%) at random empty cell (DEC-0004)

    Args:
        n_games: Number of parallel games to run
        device: PyTorch device (should be CUDA for GPU)
        spawn_fn: Optional deterministic spawn function for testing (DEC-0016)
    """

    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(
        self,
        n_games: int,
        device: torch.device,
        spawn_fn: Optional[SpawnFn] = None
    ):
        """Initialize the game environment.

        Args:
            n_games: Number of parallel games to run
            device: PyTorch device for tensor allocation
            spawn_fn: Optional function for deterministic tile spawning (testing)
        """
        self.n_games = n_games
        self.device = device
        self.spawn_fn = spawn_fn if spawn_fn is not None else _default_spawn_fn

        # Internal state: (N, 16, 17) one-hot boolean (DEC-0030)
        self._state: Optional[Tensor] = None

    def _create_empty_board(self) -> Tensor:
        """Create empty board with all positions set to empty (index 0).

        Returns:
            (N, 16, 17) one-hot board with all positions empty
        """
        board = torch.zeros(self.n_games, 16, 17, dtype=torch.bool, device=self.device)
        board[:, :, 0] = True  # All positions are empty
        return board

    def _spawn_tile(self, board: Tensor) -> Tuple[Tensor, Tensor]:
        """Spawn a new tile on the board.

        Args:
            board: (N, 16, 17) one-hot board

        Returns:
            (new_board, spawn_values): Updated board and log2 tile values spawned
        """
        # Get empty mask: position is empty if index 0 is True
        empty_mask = board[:, :, 0]  # (N, 16)

        # Get spawn positions and values from spawn function
        positions, values = self.spawn_fn(empty_mask)

        # Update board: clear old value (set to 0) and set new value
        new_board = board.clone()

        # Create indices for scatter
        batch_idx = torch.arange(self.n_games, device=self.device)

        # Clear the empty flag at spawn position
        new_board[batch_idx, positions, 0] = False

        # Set the new tile value
        new_board[batch_idx, positions, values] = True

        return new_board, values

    def _check_terminal(self, board: Tensor) -> Tensor:
        """Check if games are terminal (no valid moves).

        Args:
            board: (N, 16, 17) one-hot board

        Returns:
            (N,) boolean tensor indicating terminal games
        """
        valid_mask = compute_valid_mask(board, self.device)
        return ~valid_mask.any(dim=1)  # Terminal if no valid actions

    def reset(self) -> Tensor:
        """Reset all games to initial state.

        Returns:
            Initial board states with shape (N, 16, 17)
            Each board starts with 2 random tiles.
        """
        # Create empty boards
        self._state = self._create_empty_board()

        # Spawn 2 initial tiles
        self._state, _ = self._spawn_tile(self._state)
        self._state, _ = self._spawn_tile(self._state)

        return self._state.clone()

    def _generate_reset_states(self) -> Tensor:
        """Generate fresh reset states for episode boundaries.

        Returns:
            (N, 16, 17) fresh boards with 2 tiles each
        """
        # Create empty boards
        reset_boards = torch.zeros(self.n_games, 16, 17, dtype=torch.bool, device=self.device)
        reset_boards[:, :, 0] = True  # All positions empty

        # Spawn 2 tiles
        empty_mask = reset_boards[:, :, 0]
        positions1, values1 = self.spawn_fn(empty_mask)

        batch_idx = torch.arange(self.n_games, device=self.device)
        reset_boards[batch_idx, positions1, 0] = False
        reset_boards[batch_idx, positions1, values1] = True

        # Update empty mask and spawn second tile
        empty_mask = reset_boards[:, :, 0]
        positions2, values2 = self.spawn_fn(empty_mask)

        reset_boards[batch_idx, positions2, 0] = False
        reset_boards[batch_idx, positions2, values2] = True

        return reset_boards

    def step(self, actions: Tensor) -> StepResult:
        """Execute actions for all games.

        Args:
            actions: (N,) tensor of integers 0-3 representing moves

        Returns:
            StepResult containing:
            - next_state: (N, 16, 17) new board states
            - done: (N,) boolean game-over flags
            - merge_reward: (N,) sum of merged tile values
            - spawn_reward: (N,) value of spawned tiles
            - valid_mask: (N, 4) valid action mask for next state
            - reset_states: (N, 16, 17) fresh boards for done games

        Raises:
            InvalidMoveError: If ALL games have invalid move (DEC-0014, DEC-0027)
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        # Execute moves
        new_boards, merge_scores, move_valid = execute_moves_batched(
            self._state, actions, self.device
        )

        # Check if ALL moves are invalid (DEC-0027)
        if not move_valid.any():
            raise InvalidMoveError(
                f"All {self.n_games} games have invalid moves for actions {actions.tolist()}"
            )

        # For games with invalid moves, keep old state
        # For games with valid moves, use new state
        next_state = torch.where(
            move_valid.unsqueeze(-1).unsqueeze(-1),
            new_boards,
            self._state
        )

        # Spawn new tiles only for valid moves
        spawn_values = torch.zeros(self.n_games, dtype=torch.long, device=self.device)

        if move_valid.any():
            # Get empty positions for valid games
            empty_mask = next_state[:, :, 0]  # (N, 16)

            # Spawn tiles
            positions, values = self.spawn_fn(empty_mask)

            # Only apply spawn to games with valid moves
            batch_idx = torch.arange(self.n_games, device=self.device)

            # Create spawned board
            spawned_state = next_state.clone()
            spawned_state[batch_idx, positions, 0] = False
            spawned_state[batch_idx, positions, values] = True

            # Apply spawn only to valid games
            next_state = torch.where(
                move_valid.unsqueeze(-1).unsqueeze(-1),
                spawned_state,
                next_state
            )

            spawn_values = torch.where(move_valid, values, spawn_values)

        # Update internal state
        self._state = next_state.clone()

        # Check for terminal states
        done = self._check_terminal(next_state)

        # Compute valid mask for next state
        valid_mask = compute_valid_mask(next_state, self.device)

        # Generate reset states for episode boundaries
        reset_states = self._generate_reset_states()

        # Convert spawn values to actual tile values (2 or 4)
        spawn_reward = torch.where(
            spawn_values > 0,
            (1 << spawn_values).to(torch.int32),
            torch.zeros(self.n_games, dtype=torch.int32, device=self.device)
        )

        return StepResult(
            next_state=next_state,
            done=done,
            merge_reward=merge_scores,
            spawn_reward=spawn_reward,
            valid_mask=valid_mask,
            reset_states=reset_states,
        )
