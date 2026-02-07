"""
DQN Algorithm Module.

Milestone 3 implementation of Deep Q-Network for 2048.

Key decisions:
- DEC-0033: Uses merge_reward only
- DEC-0034: Mask-based action selection
- DEC-0035: Linear epsilon decay over 100k steps
- DEC-0036: Hard target network updates
"""

from algorithms.dqn.run import train, evaluate, TrainingResult, EvalResult
from algorithms.dqn.agent import DQNAgent
from algorithms.dqn.model import DQNNetwork
from algorithms.dqn.replay_buffer import ReplayBuffer

__all__ = [
    "train",
    "evaluate",
    "TrainingResult",
    "EvalResult",
    "DQNAgent",
    "DQNNetwork",
    "ReplayBuffer",
]
