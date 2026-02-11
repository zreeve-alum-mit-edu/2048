"""
Study Factory for Milestone 26 Full Sweep.

Generates all 200 study configurations for the experimental matrix:
- 20 algorithms
- 5 representations
- 2 reward types

Per DEC-0009: Experimental matrix is (Algorithm, Representation, Reward) tuples.
Per DEC-0011: Each (Algorithm, Representation, Reward) combo gets own study.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


# All implemented algorithms (20 total)
ALGORITHMS = [
    "dqn",
    "double_dqn",
    "dueling_dqn",
    "per_dqn",
    "nstep_dqn",
    "rainbow_lite",
    "reinforce",
    "a2c",
    "a3c",
    "ppo_gae",
    "ppo_value_clip",
    "acer",
    "impala",
    "sarsa",
    "expected_sarsa",
    "sarsa_lambda",
    "c51",
    "qr_dqn",
    "mcts_learned",
    "muzero_style",
]

# All representation types (5 total)
REPRESENTATIONS = [
    "onehot",
    "embedding",
    "cnn_2x2",
    "cnn_4x1",
    "cnn_multi",
]

# All reward types (2 total)
REWARD_TYPES = ["merge", "spawn"]


@dataclass
class SweepStudyConfig:
    """Configuration for a single study in the sweep.

    Attributes:
        study_name: Unique name for the study
        algorithm: Algorithm name
        representation: Representation type
        reward_type: Reward signal type
        n_trials: Number of Optuna trials
        epochs_per_trial: Training epochs per trial
        steps_per_epoch: Training steps per epoch
        eval_games_per_epoch: Evaluation games per epoch
        n_parallel_trials: Parallel trial count (within study)
        storage_path: SQLite storage path
    """
    study_name: str
    algorithm: str
    representation: str
    reward_type: Literal["merge", "spawn"]
    n_trials: int = 50
    epochs_per_trial: int = 300
    steps_per_epoch: int = 2500
    eval_games_per_epoch: int = 50
    n_parallel_trials: int = 1  # Sequential within study for GPU memory
    storage_path: str = "sqlite:///data/optuna/sweep.db"


class StudyFactory:
    """Factory for generating study configurations.

    Generates the full experimental matrix of studies for Milestone 26.
    """

    def __init__(
        self,
        n_trials: int = 50,
        epochs_per_trial: int = 300,
        steps_per_epoch: int = 2500,
        eval_games_per_epoch: int = 50,
        n_parallel_trials: int = 1,
        storage_path: str = "sqlite:///data/optuna/sweep.db",
        algorithms: Optional[List[str]] = None,
        representations: Optional[List[str]] = None,
        reward_types: Optional[List[str]] = None,
    ):
        """Initialize study factory.

        Args:
            n_trials: Trials per study
            epochs_per_trial: Epochs per trial
            steps_per_epoch: Steps per epoch
            eval_games_per_epoch: Eval games per epoch
            n_parallel_trials: Parallel trials within each study
            storage_path: Optuna storage path
            algorithms: Override algorithm list (default: all 20)
            representations: Override representation list (default: all 5)
            reward_types: Override reward types (default: both)
        """
        self.n_trials = n_trials
        self.epochs_per_trial = epochs_per_trial
        self.steps_per_epoch = steps_per_epoch
        self.eval_games_per_epoch = eval_games_per_epoch
        self.n_parallel_trials = n_parallel_trials
        self.storage_path = storage_path

        self.algorithms = algorithms or ALGORITHMS
        self.representations = representations or REPRESENTATIONS
        self.reward_types = reward_types or REWARD_TYPES

    def generate_all_configs(self) -> List[SweepStudyConfig]:
        """Generate all study configurations.

        Returns:
            List of SweepStudyConfig for all algorithm x repr x reward combos
        """
        configs = []

        for algo in self.algorithms:
            for repr_type in self.representations:
                for reward in self.reward_types:
                    study_name = f"{algo}_{repr_type}_{reward}"

                    config = SweepStudyConfig(
                        study_name=study_name,
                        algorithm=algo,
                        representation=repr_type,
                        reward_type=reward,
                        n_trials=self.n_trials,
                        epochs_per_trial=self.epochs_per_trial,
                        steps_per_epoch=self.steps_per_epoch,
                        eval_games_per_epoch=self.eval_games_per_epoch,
                        n_parallel_trials=self.n_parallel_trials,
                        storage_path=self.storage_path,
                    )
                    configs.append(config)

        return configs

    def get_config(self, study_name: str) -> Optional[SweepStudyConfig]:
        """Get config for a specific study.

        Args:
            study_name: Study name (e.g., "dqn_onehot_merge")

        Returns:
            SweepStudyConfig or None if not found
        """
        for config in self.generate_all_configs():
            if config.study_name == study_name:
                return config
        return None

    def get_configs_by_algorithm(self, algorithm: str) -> List[SweepStudyConfig]:
        """Get all configs for a specific algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            List of matching configs
        """
        return [c for c in self.generate_all_configs() if c.algorithm == algorithm]

    def get_configs_by_representation(self, representation: str) -> List[SweepStudyConfig]:
        """Get all configs for a specific representation.

        Args:
            representation: Representation type

        Returns:
            List of matching configs
        """
        return [c for c in self.generate_all_configs() if c.representation == representation]

    @property
    def total_studies(self) -> int:
        """Get total number of studies.

        Returns:
            Total study count
        """
        return len(self.algorithms) * len(self.representations) * len(self.reward_types)

    def summary(self) -> Dict[str, any]:
        """Get factory summary.

        Returns:
            Summary dictionary
        """
        return {
            "algorithms": self.algorithms,
            "representations": self.representations,
            "reward_types": self.reward_types,
            "total_studies": self.total_studies,
            "n_trials_per_study": self.n_trials,
            "epochs_per_trial": self.epochs_per_trial,
            "steps_per_epoch": self.steps_per_epoch,
            "total_trials": self.total_studies * self.n_trials,
        }


def get_algorithm_category(algorithm: str) -> str:
    """Get the tier/category of an algorithm.

    Args:
        algorithm: Algorithm name

    Returns:
        Category string (tier_1, tier_2, tier_3, tier_4)
    """
    tier_1 = {"dqn", "double_dqn", "reinforce", "a2c"}
    tier_2 = {"ppo_gae", "ppo_value_clip", "dueling_dqn", "per_dqn", "nstep_dqn", "rainbow_lite"}
    tier_3 = {"a3c", "acer", "impala", "sarsa", "expected_sarsa", "sarsa_lambda"}
    tier_4 = {"c51", "qr_dqn", "mcts_learned", "muzero_style"}

    if algorithm in tier_1:
        return "tier_1"
    elif algorithm in tier_2:
        return "tier_2"
    elif algorithm in tier_3:
        return "tier_3"
    elif algorithm in tier_4:
        return "tier_4"
    else:
        return "unknown"


def get_algorithm_family(algorithm: str) -> str:
    """Get the family/type of an algorithm.

    Args:
        algorithm: Algorithm name

    Returns:
        Family string (value_based, policy_gradient, actor_critic, etc.)
    """
    value_based = {"dqn", "double_dqn", "dueling_dqn", "per_dqn", "nstep_dqn",
                   "rainbow_lite", "sarsa", "expected_sarsa", "sarsa_lambda",
                   "c51", "qr_dqn"}
    policy_gradient = {"reinforce"}
    actor_critic = {"a2c", "a3c", "ppo_gae", "ppo_value_clip", "acer", "impala"}
    model_based = {"mcts_learned", "muzero_style"}

    if algorithm in value_based:
        return "value_based"
    elif algorithm in policy_gradient:
        return "policy_gradient"
    elif algorithm in actor_critic:
        return "actor_critic"
    elif algorithm in model_based:
        return "model_based"
    else:
        return "unknown"
