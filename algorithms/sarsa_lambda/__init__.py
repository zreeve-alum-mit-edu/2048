"""
SARSA(lambda) Algorithm Module with Eligibility Traces.

Milestone 21: SARSA(lambda) with one-hot representation and merge_reward.

SARSA(lambda) uses eligibility traces for multi-step credit assignment,
bridging the gap between TD(0) and Monte Carlo methods. Lambda controls
the decay rate of traces:
- lambda=0: TD(0), single-step updates
- lambda=1: Monte Carlo, full episode returns

Per DEC-0005: Algorithm modules MUST be self-contained.
"""

from algorithms.sarsa_lambda.run import train, evaluate, TrainingResult, EvalResult
from algorithms.sarsa_lambda.agent import SARSALambdaAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "SARSALambdaAgent"]
