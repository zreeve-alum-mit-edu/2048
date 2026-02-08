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
from typing import Tuple


# Fail fast if CUDA is not available (DEC-0001: target hardware is GH200)
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required but not available. "
        "This project targets NVIDIA GH200 hardware (DEC-0001). "
        "Please ensure CUDA is properly installed and a GPU is available."
    )

# Default device for lookup tables - always CUDA
_LOOKUP_TABLE_DEVICE = torch.device("cuda")


def _generate_tables(device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate all lookup tables using vectorized tensor operations (DEC-0039).

    This function generates tables for all 17^4 = 83,521 possible line configurations
    using fully vectorized PyTorch operations. No Python loops are used.

    The merge logic implements the merge-once rule (DEC-0015):
    - Each tile merges at most once per move
    - [2,2,2,2] -> [4,4,0,0], NOT [8,0,0,0]
    - Merges happen left-to-right for left move

    Args:
        device: PyTorch device to generate tables on (should be CUDA)

    Returns:
        (line_transition, valid_move, score_delta) tensors on the specified device
    """
    # Step 1: Generate all 83,521 input configurations as (83521, 4) tensor
    # Using meshgrid to create all combinations of [0..16] x [0..16] x [0..16] x [0..16]
    vals = torch.arange(17, dtype=torch.int32, device=device)
    t0, t1, t2, t3 = torch.meshgrid(vals, vals, vals, vals, indexing='ij')
    # Flatten to (83521, 4) - each row is one line configuration
    inputs = torch.stack([t0.flatten(), t1.flatten(), t2.flatten(), t3.flatten()], dim=1)
    n_configs = inputs.shape[0]  # 83521

    # Step 2: Compact non-zeros to left using parallel sort/gather
    # We need to move all non-zero values to the left while preserving order
    # Strategy: create a sort key that puts zeros last while preserving relative order
    is_nonzero = inputs > 0  # (83521, 4) bool
    # Count non-zeros per line for later use
    nonzero_counts = is_nonzero.sum(dim=1)  # (83521,)

    # Create position indices
    positions = torch.arange(4, device=device).unsqueeze(0).expand(n_configs, -1)  # (83521, 4)

    # Sort key: zeros get position + 4, non-zeros keep position
    # This pushes zeros to the end while maintaining order of non-zeros
    sort_keys = torch.where(is_nonzero, positions, positions + 4)
    sorted_indices = sort_keys.argsort(dim=1, stable=True)  # (83521, 4)

    # Gather compacted values
    compacted = torch.gather(inputs, 1, sorted_indices)  # (83521, 4)
    c0, c1, c2, c3 = compacted[:, 0], compacted[:, 1], compacted[:, 2], compacted[:, 3]

    # Step 3: Apply merge rules via tensor masks
    # After compaction: [c0, c1, c2, c3] where zeros are at the end
    # Possible merge patterns for left move (left-to-right):
    #   Pattern A: c0==c1 merges, then check c2==c3
    #   Pattern B: c0!=c1, then c1==c2 merges, c3 stays
    #   Pattern C: c0!=c1, c1!=c2, then c2==c3 merges
    #   Pattern D: no merges possible

    # First merge: positions 0,1
    merge_01 = (c0 == c1) & (c0 > 0)  # (83521,) bool

    # After potential merge at 0,1:
    # If merge_01: remaining to check is [c2, c3]
    # If not merge_01: remaining to check is [c1, c2, c3]

    # Second merge depends on first merge:
    # If merge_01: check c2==c3
    merge_23_after_01 = merge_01 & (c2 == c3) & (c2 > 0)

    # If not merge_01: check c1==c2
    merge_12 = (~merge_01) & (c1 == c2) & (c1 > 0)

    # If not merge_01 and not merge_12: check c2==c3
    merge_23_no_prior = (~merge_01) & (~merge_12) & (c2 == c3) & (c2 > 0)

    # Step 4: Compute outputs using torch.where with conditional masks
    # Initialize output slots
    out0 = torch.zeros(n_configs, dtype=torch.int32, device=device)
    out1 = torch.zeros(n_configs, dtype=torch.int32, device=device)
    out2 = torch.zeros(n_configs, dtype=torch.int32, device=device)
    out3 = torch.zeros(n_configs, dtype=torch.int32, device=device)

    # Case 1: merge_01 AND merge_23_after_01 (two merges: [m01, m23, 0, 0])
    case1 = merge_01 & merge_23_after_01
    out0 = torch.where(case1, c0 + 1, out0)
    out1 = torch.where(case1, c2 + 1, out1)
    # out2, out3 stay 0

    # Case 2: merge_01 AND NOT merge_23_after_01 (one merge at 0,1: [m01, c2, c3, 0])
    case2 = merge_01 & (~merge_23_after_01)
    out0 = torch.where(case2, c0 + 1, out0)
    out1 = torch.where(case2, c2, out1)
    out2 = torch.where(case2, c3, out2)
    # out3 stays 0

    # Case 3: NOT merge_01 AND merge_12 (one merge at 1,2: [c0, m12, c3, 0])
    case3 = (~merge_01) & merge_12
    out0 = torch.where(case3, c0, out0)
    out1 = torch.where(case3, c1 + 1, out1)
    out2 = torch.where(case3, c3, out2)
    # out3 stays 0

    # Case 4: NOT merge_01 AND NOT merge_12 AND merge_23_no_prior (one merge at 2,3: [c0, c1, m23, 0])
    case4 = (~merge_01) & (~merge_12) & merge_23_no_prior
    out0 = torch.where(case4, c0, out0)
    out1 = torch.where(case4, c1, out1)
    out2 = torch.where(case4, c2 + 1, out2)
    # out3 stays 0

    # Case 5: No merges at all (keep compacted: [c0, c1, c2, c3])
    case5 = (~merge_01) & (~merge_12) & (~merge_23_no_prior)
    out0 = torch.where(case5, c0, out0)
    out1 = torch.where(case5, c1, out1)
    out2 = torch.where(case5, c2, out2)
    out3 = torch.where(case5, c3, out3)

    # Stack outputs
    outputs = torch.stack([out0, out1, out2, out3], dim=1)  # (83521, 4)

    # Step 5: Compute score delta
    # Score is sum of merged tile values (2^merged_value for each merge)
    score = torch.zeros(n_configs, dtype=torch.int32, device=device)

    # Merge at 0,1 contributes 2^(c0+1)
    score = torch.where(merge_01, score + (1 << (c0 + 1).clamp(max=30)).to(torch.int32), score)

    # Merge at 2,3 after 0,1 contributes 2^(c2+1)
    score = torch.where(merge_23_after_01, score + (1 << (c2 + 1).clamp(max=30)).to(torch.int32), score)

    # Merge at 1,2 (when no 0,1 merge) contributes 2^(c1+1)
    score = torch.where(merge_12, score + (1 << (c1 + 1).clamp(max=30)).to(torch.int32), score)

    # Merge at 2,3 (when no prior merges) contributes 2^(c2+1)
    score = torch.where(merge_23_no_prior, score + (1 << (c2 + 1).clamp(max=30)).to(torch.int32), score)

    # Step 6: Compute valid move (did line change?)
    # Line changed if outputs != inputs
    valid = (outputs != inputs).any(dim=1)  # (83521,) bool

    # Step 7: Reshape to final shapes
    shape_4d = (17, 17, 17, 17)

    # Valid move: (17, 17, 17, 17) boolean
    valid_move = valid.reshape(shape_4d)

    # Score delta: (17, 17, 17, 17) int32
    score_delta = score.reshape(shape_4d)

    # Line transition: (17, 17, 17, 17, 4, 18) boolean one-hot
    # Create indices for scatter
    line_transition = torch.zeros(17, 17, 17, 17, 4, 18, dtype=torch.bool, device=device)

    # We need to set line_transition[t0, t1, t2, t3, pos, output_val] = True
    # Use advanced indexing with flat indices
    flat_idx = torch.arange(n_configs, device=device)
    idx_t0 = inputs[:, 0].long()
    idx_t1 = inputs[:, 1].long()
    idx_t2 = inputs[:, 2].long()
    idx_t3 = inputs[:, 3].long()

    # Set one-hot for each output position
    line_transition[idx_t0, idx_t1, idx_t2, idx_t3, 0, outputs[:, 0].long()] = True
    line_transition[idx_t0, idx_t1, idx_t2, idx_t3, 1, outputs[:, 1].long()] = True
    line_transition[idx_t0, idx_t1, idx_t2, idx_t3, 2, outputs[:, 2].long()] = True
    line_transition[idx_t0, idx_t1, idx_t2, idx_t3, 3, outputs[:, 3].long()] = True

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
