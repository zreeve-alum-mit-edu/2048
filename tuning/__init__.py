"""
Hyperparameter Tuning Module.

This module provides Optuna integration for hyperparameter tuning
of DQN across different representations and reward types.

Per DEC-0037: Milestone 5 Spec Packet approved.
Per DEC-0010: Uses Optuna with SQLite storage.
Per DEC-0011: Each (Algorithm, Representation, Reward) combo gets own study.
"""

from tuning.study_config import StudyConfig, STUDY_CONFIGS
from tuning.utils import create_representation
from tuning.search_spaces import suggest_hyperparams

__all__ = [
    "StudyConfig",
    "STUDY_CONFIGS",
    "create_representation",
    "suggest_hyperparams",
]
