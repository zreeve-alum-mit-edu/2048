"""
n-step DQN Network Model.

Reuses the same network architecture as Double DQN.
n-step is about return computation, not network architecture.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

# Reuse Double DQN model
from algorithms.double_dqn.model import DoubleDQNNetwork

__all__ = ['DoubleDQNNetwork']
