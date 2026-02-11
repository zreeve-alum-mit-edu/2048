"""
PPO+GAE (Proximal Policy Optimization with Generalized Advantage Estimation).

Milestone 10: Tier 2 RL Algorithm.
"""

from algorithms.ppo_gae.run import train, evaluate, TrainingResult, EvalResult
from algorithms.ppo_gae.agent import PPOAgent

__all__ = ['train', 'evaluate', 'TrainingResult', 'EvalResult', 'PPOAgent']
