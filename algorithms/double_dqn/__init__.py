"""
Double DQN Algorithm Module.

Per DEC-0005: Algorithm modules are self-contained in algorithms/<name>/
This module implements Double DQN (Milestone 9) - DQN with target network
improvement to reduce overestimation bias.
"""

from algorithms.double_dqn.run import train, evaluate, TrainingResult, EvalResult
from algorithms.double_dqn.agent import DoubleDQNAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "DoubleDQNAgent"]
