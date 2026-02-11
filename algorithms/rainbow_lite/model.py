"""
Rainbow-lite Network Model.

Uses the Dueling DQN architecture with:
- Shared trunk for feature extraction
- Separate value and advantage streams

The "lite" version excludes:
- NoisyNet (uses standard layers with epsilon-greedy)
- C51/distributional (uses standard Q-values)

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

# Reuse Dueling DQN model
from algorithms.dueling_dqn.model import DuelingDQNNetwork

__all__ = ['DuelingDQNNetwork']
