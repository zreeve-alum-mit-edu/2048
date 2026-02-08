"""
Precomputed lookup tables for GPU-native 2048 game operations.

Tables are generated once at module import time (DEC-0028) directly on GPU.
They support O(1) lookups for line transitions, valid moves, and score deltas.

CUDA Requirement: This module REQUIRES CUDA to be available. It will fail fast
at import time if CUDA is not available, per the target hardware decision
(DEC-0001: Target hardware is NVIDIA GH200).

Table shapes and dtypes (DEC-0029):
- LINE_TRANSITION: (17, 17, 17, 17, 4, 18) boolean - one-hot encoded output line
  (18 values: 0=empty, 1-17 = 2^1 to 2^17; index 17 for merged 65536+65536)
- VALID_MOVE: (17, 17, 17, 17) boolean - whether left move is valid
- SCORE_DELTA: (17, 17, 17, 17) int32 - points earned from merges

All tables are computed for the LEFT direction only. Other directions
are handled by rotating/transforming the board before lookup.
"""

import torch
from torch import Tensor
from typing import Tuple, List


# Fail fast if CUDA is not available (DEC-0001: target hardware is GH200)
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required but not available. "
        "This project targets NVIDIA GH200 hardware (DEC-0001). "
        "Please ensure CUDA is properly installed and a GPU is available."
    )

# Default device for lookup tables - always CUDA
_LOOKUP_TABLE_DEVICE = torch.device("cuda")


def _compute_line_left(line: List[int]) -> Tuple[List[int], int]:
    """Compute result of LEFT move on a 4-element line.

    Implements merge-once rule (DEC-0015):
    - Each tile merges at most once per move
    - [2,2,2,2] -> [4,4,0,0], NOT [8,0,0,0]
    - Merges happen left-to-right for left move

    Args:
        line: 4-element list of log2-encoded values (0=empty, 1=2, 2=4, etc.)

    Returns:
        (result_line, score): Result line and merge score
    """
    # First: collect non-empty tiles
    non_empty = [v for v in line if v > 0]

    if len(non_empty) == 0:
        return [0, 0, 0, 0], 0

    # Process merges left-to-right
    result = []
    score = 0
    i = 0

    while i < len(non_empty):
        if i + 1 < len(non_empty) and non_empty[i] == non_empty[i + 1]:
            # Merge: tiles combine, value increases by 1 (in log2 space)
            merged_value = non_empty[i] + 1
            result.append(merged_value)
            # Score is the actual tile value (2^merged_value)
            score += (1 << merged_value)
            i += 2  # Skip both merged tiles
        else:
            result.append(non_empty[i])
            i += 1

    # Pad with zeros
    while len(result) < 4:
        result.append(0)

    return result, score


def _line_changed(original: List[int], result: List[int]) -> bool:
    """Check if line changed after move."""
    return original != result


def _generate_tables(device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate all lookup tables directly on the specified device.

    Args:
        device: PyTorch device to generate tables on (should be CUDA)

    Returns:
        (line_transition, valid_move, score_delta) tensors on the specified device
    """
    # Line transition: (17, 17, 17, 17, 4, 18) boolean one-hot
    # For each input line (4 tiles, each 0-16), output is one-hot for each position
    # Output can be 0-17 (merging two 16s produces 17)
    line_transition = torch.zeros(17, 17, 17, 17, 4, 18, dtype=torch.bool, device=device)

    # Valid move: (17, 17, 17, 17) boolean
    valid_move = torch.zeros(17, 17, 17, 17, dtype=torch.bool, device=device)

    # Score delta: (17, 17, 17, 17) int32
    score_delta = torch.zeros(17, 17, 17, 17, dtype=torch.int32, device=device)

    # Iterate all possible line configurations
    for t0 in range(17):
        for t1 in range(17):
            for t2 in range(17):
                for t3 in range(17):
                    input_line = [t0, t1, t2, t3]
                    output_line, score = _compute_line_left(input_line)

                    # Store one-hot output
                    for pos in range(4):
                        line_transition[t0, t1, t2, t3, pos, output_line[pos]] = True

                    # Store validity (did line change?)
                    valid_move[t0, t1, t2, t3] = _line_changed(input_line, output_line)

                    # Store score
                    score_delta[t0, t1, t2, t3] = score

    return line_transition, valid_move, score_delta


# Generate tables at import time directly on GPU (DEC-0028)
LINE_TRANSITION, VALID_MOVE, SCORE_DELTA = _generate_tables(_LOOKUP_TABLE_DEVICE)


def get_tables_on_device(device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """Get lookup tables on specified device.

    Tables are generated on CUDA at import time and returned directly for CUDA
    devices. For other devices (e.g., CPU for testing), tables are moved and cached.

    Note: Since target hardware is GH200 (DEC-0001), the primary path returns
    the pre-generated CUDA tables directly without any device transfer.

    Args:
        device: Target PyTorch device

    Returns:
        (line_transition, valid_move, score_delta) on device
    """
    # Fast path: if requesting CUDA (same device as tables), return directly
    if device.type == "cuda":
        # Handle cuda vs cuda:0 vs cuda:N - all return the same tables
        # since we only support single GPU operation
        return LINE_TRANSITION, VALID_MOVE, SCORE_DELTA

    # Slow path: for non-CUDA devices (e.g., CPU for testing), cache transfers
    if not hasattr(get_tables_on_device, '_cache'):
        get_tables_on_device._cache = {}

    cache_key = str(device)
    if cache_key not in get_tables_on_device._cache:
        get_tables_on_device._cache[cache_key] = (
            LINE_TRANSITION.to(device),
            VALID_MOVE.to(device),
            SCORE_DELTA.to(device),
        )

    return get_tables_on_device._cache[cache_key]
