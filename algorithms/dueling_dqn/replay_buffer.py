"""
Experience Replay Buffer for Dueling DQN.

Reuses the same replay buffer implementation as Double DQN.

Per DEC-0003: Replay buffers MUST NOT contain cross-episode transitions.
"""

# Reuse Double DQN replay buffer - identical functionality
from algorithms.double_dqn.replay_buffer import ReplayBuffer, Transition

__all__ = ['ReplayBuffer', 'Transition']
