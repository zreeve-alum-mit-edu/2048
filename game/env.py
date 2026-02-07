"""
GPU-native 2048 Game Environment.

This module provides the GameEnv class for running N parallel 2048 games
on GPU using PyTorch tensors.

NOTE: This is a stub implementation for test-first development (Milestone 1).
      Full implementation will be completed in Milestone 2.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


class InvalidMoveError(Exception):
    """Raised when an action causes no board change.

    Per DEC-0014: GameEnv MUST raise exception on invalid move.
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

        Raises:
            NotImplementedError: Stub - implementation in Milestone 2
        """
        raise NotImplementedError("GameEnv not yet implemented - Milestone 2")

    def reset(self) -> Tensor:
        """Reset all games to initial state.

        Returns:
            Initial board states with shape (N, 16, 17)
            Each board starts with 2 random tiles.

        Raises:
            NotImplementedError: Stub - implementation in Milestone 2
        """
        raise NotImplementedError("GameEnv.reset() not yet implemented - Milestone 2")

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
            InvalidMoveError: If any action causes no board change (DEC-0014)
            NotImplementedError: Stub - implementation in Milestone 2
        """
        raise NotImplementedError("GameEnv.step() not yet implemented - Milestone 2")
