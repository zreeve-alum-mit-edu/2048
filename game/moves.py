"""
Move execution logic for GPU-native 2048 game.

This module provides functions to execute moves on batched game states
using precomputed lookup tables for O(1) per-line operations.

All operations work on one-hot encoded boards (N, 16, 17).
"""

import torch
from torch import Tensor
from typing import Tuple

from game.lookup_tables import get_tables_on_device


# Action constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


def _get_line_indices(board: Tensor) -> Tensor:
    """Convert one-hot board to integer indices for table lookup.

    Args:
        board: (N, 16, 17) one-hot encoded board

    Returns:
        (N, 16) tensor of integer tile values (0-16)
    """
    # Convert bool to int for argmax (bool not supported)
    return board.to(torch.int64).argmax(dim=-1)


def _apply_line_transition_left(
    line_indices: Tensor,
    line_transition: Tensor,
    score_delta: Tensor,
    valid_move: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply LEFT move to all rows using lookup tables.

    Args:
        line_indices: (N, 16) integer tile values arranged as 4x4 board
        line_transition: Lookup table for line transitions
        score_delta: Lookup table for merge scores
        valid_move: Lookup table for move validity

    Returns:
        (new_board_onehot, total_score, row_valid): Results of the move
        - new_board_onehot: (N, 16, 17) one-hot encoded result
        - total_score: (N,) merge scores
        - row_valid: (N, 4) validity per row
    """
    n_games = line_indices.shape[0]
    device = line_indices.device

    # Reshape to (N, 4, 4) for row access: [batch, row, col]
    board_2d = line_indices.view(n_games, 4, 4)

    # Extract indices for each position in each row
    t0 = board_2d[:, :, 0]  # (N, 4) - first column of each row
    t1 = board_2d[:, :, 1]  # (N, 4)
    t2 = board_2d[:, :, 2]  # (N, 4)
    t3 = board_2d[:, :, 3]  # (N, 4)

    # Lookup transition for each row: (N, 4, 4, 18)
    # line_transition[t0, t1, t2, t3] gives (N, 4, 4, 18) one-hot output
    # (18 values to handle merging two 65536 tiles -> 131072)
    new_rows_onehot = line_transition[t0, t1, t2, t3]  # (N, 4, 4, 18)

    # Truncate to 17 values for board representation
    # Index 17 (131072 = 2^17) is extremely rare and capped at 16
    new_rows_onehot = new_rows_onehot[:, :, :, :17]  # (N, 4, 4, 17)

    # Reshape to (N, 16, 17)
    new_board_onehot = new_rows_onehot.view(n_games, 16, 17)

    # Lookup scores for each row: (N, 4)
    row_scores = score_delta[t0, t1, t2, t3]  # (N, 4)
    total_score = row_scores.sum(dim=1).to(torch.int32)  # (N,) ensure int32

    # Lookup validity for each row: (N, 4)
    row_valid = valid_move[t0, t1, t2, t3]  # (N, 4)

    return new_board_onehot, total_score, row_valid


def _transpose_board(board: Tensor) -> Tensor:
    """Transpose board (swap rows and columns).

    Args:
        board: (N, 16, 17) one-hot board

    Returns:
        Transposed (N, 16, 17) board
    """
    n_games = board.shape[0]
    # Reshape to (N, 4, 4, 17)
    board_2d = board.view(n_games, 4, 4, 17)
    # Transpose spatial dims (swap rows and cols)
    board_2d = board_2d.transpose(1, 2)
    return board_2d.reshape(n_games, 16, 17)


def _reverse_rows(board: Tensor) -> Tensor:
    """Reverse each row in the board.

    Args:
        board: (N, 16, 17) one-hot board

    Returns:
        Board with reversed rows (N, 16, 17)
    """
    n_games = board.shape[0]
    # Reshape to (N, 4, 4, 17)
    board_2d = board.view(n_games, 4, 4, 17)
    # Flip along column dimension
    board_2d = board_2d.flip(dims=[2])
    return board_2d.reshape(n_games, 16, 17)


def execute_move(
    board: Tensor,
    action: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Execute a single action on all boards.

    All directions are normalized to LEFT via transformation, then the lookup
    table is applied, then transformed back.

    Action mapping:
    - LEFT (2): apply LEFT directly
    - RIGHT (3): reverse rows, apply LEFT, reverse rows
    - UP (0): transpose, apply LEFT, transpose
    - DOWN (1): transpose, reverse rows, apply LEFT, reverse rows, transpose

    Args:
        board: (N, 16, 17) one-hot encoded boards
        action: Single action integer (0-3)
        device: PyTorch device

    Returns:
        (new_board, scores, valid): Move results
        - new_board: (N, 16, 17) one-hot boards after move
        - scores: (N,) merge rewards
        - valid: (N,) boolean whether move was valid for each game
    """
    line_transition, valid_move, score_delta = get_tables_on_device(device)

    # Transform to normalize to LEFT
    if action == UP:
        # Transpose: columns become rows, then LEFT moves tiles "up"
        transformed = _transpose_board(board)
    elif action == DOWN:
        # Transpose + reverse: so LEFT moves tiles "down"
        transformed = _reverse_rows(_transpose_board(board))
    elif action == LEFT:
        transformed = board
    else:  # RIGHT
        # Reverse rows: so LEFT moves tiles "right"
        transformed = _reverse_rows(board)

    # Get integer indices for lookup
    line_indices = _get_line_indices(transformed)

    # Apply LEFT move via lookup
    new_board, scores, row_valid = _apply_line_transition_left(
        line_indices, line_transition, score_delta, valid_move
    )

    # Any row valid means the whole move is valid
    valid = row_valid.any(dim=1)

    # Transform back
    if action == UP:
        new_board = _transpose_board(new_board)
    elif action == DOWN:
        new_board = _transpose_board(_reverse_rows(new_board))
    elif action == RIGHT:
        new_board = _reverse_rows(new_board)
    # LEFT needs no transformation back

    return new_board, scores, valid


def execute_moves_batched(
    board: Tensor,
    actions: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Execute per-game actions on batched boards.

    Each game can have a different action.

    Args:
        board: (N, 16, 17) one-hot encoded boards
        actions: (N,) action integers per game
        device: PyTorch device

    Returns:
        (new_boards, scores, valid): Move results per game
    """
    n_games = board.shape[0]
    line_transition, valid_move, score_delta = get_tables_on_device(device)

    # Initialize output tensors
    new_boards = torch.zeros_like(board)
    scores = torch.zeros(n_games, dtype=torch.int32, device=device)
    valid = torch.zeros(n_games, dtype=torch.bool, device=device)

    # Process each action type separately for efficiency
    for action in range(4):
        mask = (actions == action)
        if not mask.any():
            continue

        action_boards = board[mask]
        result_board, result_scores, result_valid = execute_move(
            action_boards, action, device
        )

        new_boards[mask] = result_board
        scores[mask] = result_scores
        valid[mask] = result_valid

    return new_boards, scores, valid


def compute_valid_mask(board: Tensor, device: torch.device) -> Tensor:
    """Compute valid action mask for all boards.

    Args:
        board: (N, 16, 17) one-hot encoded boards
        device: PyTorch device

    Returns:
        (N, 4) boolean mask of valid actions [UP, DOWN, LEFT, RIGHT]
    """
    n_games = board.shape[0]
    valid_mask = torch.zeros(n_games, 4, dtype=torch.bool, device=device)

    for action in range(4):
        _, _, valid = execute_move(board, action, device)
        valid_mask[:, action] = valid

    return valid_mask
