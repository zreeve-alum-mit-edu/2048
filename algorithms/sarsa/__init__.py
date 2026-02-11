"""
SARSA (State-Action-Reward-State-Action) Algorithm Module.

Milestone 19: SARSA with one-hot representation and merge_reward.

SARSA is an on-policy TD control algorithm. Unlike Q-learning which
uses max_a Q(s',a), SARSA uses the actual next action taken under
the current policy: Q(s,a) += alpha * (r + gamma*Q(s',a') - Q(s,a))

Per DEC-0005: Algorithm modules MUST be self-contained.
"""

from algorithms.sarsa.run import train, evaluate, TrainingResult, EvalResult
from algorithms.sarsa.agent import SARSAAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "SARSAAgent"]
