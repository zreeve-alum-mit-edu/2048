"""
PER DQN Network Model.

Reuses the same network architecture as Double DQN.
PER is about sampling strategy, not network architecture.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

# Reuse Double DQN model - architecture is identical
from algorithms.double_dqn.model import DoubleDQNNetwork

__all__ = ['DoubleDQNNetwork']
