"""
A3C (Asynchronous Advantage Actor-Critic) Algorithm Module.

Milestone 16: A3C with one-hot representation and merge_reward.

A3C extends A2C with asynchronous parallel workers that each interact
with their own environment and compute gradients locally. This
implementation simulates async behavior with vectorized environments
for GPU efficiency while maintaining the core A3C gradient accumulation
pattern.

Per DEC-0005: Algorithm modules MUST be self-contained.
"""

from algorithms.a3c.run import train, evaluate, TrainingResult, EvalResult
from algorithms.a3c.agent import A3CAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "A3CAgent"]
