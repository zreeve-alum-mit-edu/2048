"""
Study Configuration.

Defines configuration dataclass and all 10 study configs for DQN tuning.

Per DEC-0009: Experimental matrix is (Algorithm, Representation, Reward) tuples.
Per DEC-0011: Each (Algorithm, Representation, Reward) combo gets own study.
Per DEC-0037: 50 trials, 300 epochs, 2500 steps/epoch, 4 parallel trials.
"""

from dataclasses import dataclass
from typing import Literal, Dict


@dataclass
class StudyConfig:
    """Configuration for a single Optuna study.

    Attributes:
        study_name: Unique name for the study
        representation_type: One of the 5 representation variants
        reward_type: Either "merge" or "spawn"
        n_trials: Number of trials per study (default 50)
        epochs_per_trial: Training epochs per trial (default 300)
        steps_per_epoch: Training steps per epoch (default 2500)
        eval_games_per_epoch: Evaluation games per epoch (default 50)
        n_parallel_trials: Parallel trial count (default 4)
        storage_path: SQLite storage path
    """
    study_name: str
    representation_type: Literal["onehot", "embedding", "cnn_2x2", "cnn_4x1", "cnn_multi"]
    reward_type: Literal["merge", "spawn"]
    n_trials: int = 50
    epochs_per_trial: int = 300
    steps_per_epoch: int = 2500
    eval_games_per_epoch: int = 50
    n_parallel_trials: int = 4
    storage_path: str = "sqlite:///data/optuna/dqn_tuning.db"


# Predefined study configs for all 10 studies (5 representations x 2 rewards)
STUDY_CONFIGS: Dict[str, StudyConfig] = {
    # OneHot representation
    "dqn_onehot_merge": StudyConfig(
        study_name="dqn_onehot_merge",
        representation_type="onehot",
        reward_type="merge"
    ),
    "dqn_onehot_spawn": StudyConfig(
        study_name="dqn_onehot_spawn",
        representation_type="onehot",
        reward_type="spawn"
    ),

    # Embedding representation
    "dqn_embedding_merge": StudyConfig(
        study_name="dqn_embedding_merge",
        representation_type="embedding",
        reward_type="merge"
    ),
    "dqn_embedding_spawn": StudyConfig(
        study_name="dqn_embedding_spawn",
        representation_type="embedding",
        reward_type="spawn"
    ),

    # CNN-2x2 representation
    "dqn_cnn2x2_merge": StudyConfig(
        study_name="dqn_cnn2x2_merge",
        representation_type="cnn_2x2",
        reward_type="merge"
    ),
    "dqn_cnn2x2_spawn": StudyConfig(
        study_name="dqn_cnn2x2_spawn",
        representation_type="cnn_2x2",
        reward_type="spawn"
    ),

    # CNN-4x1 representation (row/column kernels)
    "dqn_cnn4x1_merge": StudyConfig(
        study_name="dqn_cnn4x1_merge",
        representation_type="cnn_4x1",
        reward_type="merge"
    ),
    "dqn_cnn4x1_spawn": StudyConfig(
        study_name="dqn_cnn4x1_spawn",
        representation_type="cnn_4x1",
        reward_type="spawn"
    ),

    # CNN-Multi representation (Inception-style)
    "dqn_cnnmulti_merge": StudyConfig(
        study_name="dqn_cnnmulti_merge",
        representation_type="cnn_multi",
        reward_type="merge"
    ),
    "dqn_cnnmulti_spawn": StudyConfig(
        study_name="dqn_cnnmulti_spawn",
        representation_type="cnn_multi",
        reward_type="spawn"
    ),
}
