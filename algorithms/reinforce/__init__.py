"""
REINFORCE Algorithm Module.

Per DEC-0005: Algorithm modules are self-contained in algorithms/<name>/
This module implements REINFORCE (Milestone 7) - vanilla policy gradient
algorithm for baseline/sanity check.
"""

from algorithms.reinforce.run import train, evaluate, TrainingResult, EvalResult
from algorithms.reinforce.agent import REINFORCEAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "REINFORCEAgent"]
