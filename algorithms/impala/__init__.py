"""
IMPALA (Importance Weighted Actor-Learner Architecture) Algorithm Module.

Milestone 18: IMPALA with V-trace, one-hot representation and merge_reward.

IMPALA decouples acting from learning using V-trace importance sampling
correction to handle the policy lag between actors and the learner.

Per DEC-0005: Algorithm modules MUST be self-contained.
"""

from algorithms.impala.run import train, evaluate, TrainingResult, EvalResult
from algorithms.impala.agent import IMPALAAgent

__all__ = ["train", "evaluate", "TrainingResult", "EvalResult", "IMPALAAgent"]
