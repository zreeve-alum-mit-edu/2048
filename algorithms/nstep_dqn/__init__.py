"""
n-step DQN (Multi-step DQN).

Milestone 14: Tier 2 RL Algorithm.
Uses multi-step returns for faster value propagation.
"""

from algorithms.nstep_dqn.run import train, evaluate, TrainingResult, EvalResult
from algorithms.nstep_dqn.agent import NStepDQNAgent

__all__ = ['train', 'evaluate', 'TrainingResult', 'EvalResult', 'NStepDQNAgent']
