"""
Expected SARSA Algorithm Module.

Milestone 20: Expected SARSA with one-hot representation and merge_reward.

Expected SARSA uses the expected value of Q under the policy instead
of the sampled next action, reducing variance compared to SARSA:
Q(s,a) += alpha * (r + gamma * E[Q(s',a')] - Q(s,a))

Per DEC-0005: Algorithm modules MUST be self-contained.
"""

from algorithms.expected_sarsa.run import train, evaluate, TrainingResult, EvalResult
from algorithms.expected_sarsa.agent import ExpectedSARSAAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "ExpectedSARSAAgent"]
