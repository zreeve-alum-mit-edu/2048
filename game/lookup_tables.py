"""
Precomputed lookup tables for GPU-native 2048 game operations.

Tables are generated once at module import time (DEC-0028) and cached.
They support O(1) lookups for line transitions, valid moves, and score deltas.

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


def _generate_tables() -> Tuple[Tensor, Tensor, Tensor]:
    """Generate all lookup tables.

    Returns:
        (line_transition, valid_move, score_delta) tensors
    """
    # Line transition: (17, 17, 17, 17, 4, 18) boolean one-hot
    # For each input line (4 tiles, each 0-16), output is one-hot for each position
    # Output can be 0-17 (merging two 16s produces 17)
    line_transition = torch.zeros(17, 17, 17, 17, 4, 18, dtype=torch.bool)

    # Valid move: (17, 17, 17, 17) boolean
    valid_move = torch.zeros(17, 17, 17, 17, dtype=torch.bool)

    # Score delta: (17, 17, 17, 17) int32
    score_delta = torch.zeros(17, 17, 17, 17, dtype=torch.int32)

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


# Generate tables at import time (DEC-0028)
LINE_TRANSITION, VALID_MOVE, SCORE_DELTA = _generate_tables()


def get_tables_on_device(device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """Get lookup tables on specified device.

    Tables are lazily moved to device and cached.

    Args:
        device: Target PyTorch device

    Returns:
        (line_transition, valid_move, score_delta) on device
    """
    global _device_cache

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
