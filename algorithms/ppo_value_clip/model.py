"""
PPO+Value Clip Actor-Critic Network Model.

Reuses the same architecture as PPO+GAE. The value clipping is implemented
in the agent's loss computation, not in the network.

Per DEC-0037: Input size is dynamic based on representation output shape.
"""

# Reuse PPO+GAE model - architecture is identical
from algorithms.ppo_gae.model import PPOActorCriticNetwork

__all__ = ['PPOActorCriticNetwork']
