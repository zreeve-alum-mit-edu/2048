"""
ACER (Actor-Critic with Experience Replay) Algorithm Module.

Milestone 17: ACER with one-hot representation and merge_reward.

ACER combines on-policy actor-critic with off-policy experience replay
by using importance sampling correction (Retrace). This allows for
better sample efficiency while maintaining stability.

Per DEC-0005: Algorithm modules MUST be self-contained.
"""

from algorithms.acer.run import train, evaluate, TrainingResult, EvalResult
from algorithms.acer.agent import ACERAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "ACERAgent"]
