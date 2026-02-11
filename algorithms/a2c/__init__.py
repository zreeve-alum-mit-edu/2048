"""
A2C (Advantage Actor-Critic) Algorithm Module.

Per DEC-0005: Algorithm modules are self-contained in algorithms/<name>/
This module implements A2C (Milestone 8) - synchronous Advantage Actor-Critic
algorithm.
"""

from algorithms.a2c.run import train, evaluate, TrainingResult, EvalResult
from algorithms.a2c.agent import A2CAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "A2CAgent"]
