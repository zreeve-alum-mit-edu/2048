"""
Rainbow-lite (Simplified Rainbow DQN).

Milestone 15: Tier 2 RL Algorithm.
Combines Double DQN + Dueling architecture + PER + n-step returns.
(Excludes C51/NoisyNet from full Rainbow for simplicity)
"""

from algorithms.rainbow_lite.run import train, evaluate, TrainingResult, EvalResult
from algorithms.rainbow_lite.agent import RainbowLiteAgent

__all__ = ['train', 'evaluate', 'TrainingResult', 'EvalResult', 'RainbowLiteAgent']
