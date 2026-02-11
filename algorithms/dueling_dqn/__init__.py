"""
Dueling DQN (Dueling Network Architecture for Deep Q-Learning).

Milestone 12: Tier 2 RL Algorithm.
Separates value and advantage streams for better representation.
"""

from algorithms.dueling_dqn.run import train, evaluate, TrainingResult, EvalResult
from algorithms.dueling_dqn.agent import DuelingDQNAgent

__all__ = ['train', 'evaluate', 'TrainingResult', 'EvalResult', 'DuelingDQNAgent']
