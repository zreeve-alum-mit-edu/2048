"""
Experiment Configuration.

Defines configuration dataclasses for orchestrator experiments.

Per DEC-0009: Experimental matrix is (Algorithm, Representation, Reward) tuples.
Per Milestone 6: Config-driven experiment launching.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

import yaml


# Valid representation types per DEC-0008 and existing representations/
VALID_REPRESENTATIONS = frozenset([
    "onehot",
    "embedding",
    "cnn_2x2",
    "cnn_4x1",
    "cnn_multi",
])

# Valid reward types per design docs
VALID_REWARD_TYPES = frozenset(["merge", "spawn"])


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment.

    Defines an (Algorithm, Representation, Reward) combination with
    training parameters.

    Attributes:
        name: Unique experiment identifier
        algorithm: Algorithm name (maps to algorithms/<name>/)
        representation: Representation type (onehot, embedding, etc.)
        reward_type: Either "merge" or "spawn"
        training_steps: Number of training steps
        eval_games: Number of evaluation games after training
        n_envs: Number of parallel environments (default 32)
        hyperparameters: Optional override for algorithm hyperparameters
        checkpoint_dir: Where to save checkpoints (relative to results_dir)
    """
    name: str
    algorithm: str
    representation: str
    reward_type: Literal["merge", "spawn"]
    training_steps: int = 100000
    eval_games: int = 100
    n_envs: int = 32
    hyperparameters: Optional[Dict[str, Any]] = None
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.representation not in VALID_REPRESENTATIONS:
            raise ValueError(
                f"Invalid representation '{self.representation}'. "
                f"Must be one of: {sorted(VALID_REPRESENTATIONS)}"
            )
        if self.reward_type not in VALID_REWARD_TYPES:
            raise ValueError(
                f"Invalid reward_type '{self.reward_type}'. "
                f"Must be one of: {sorted(VALID_REWARD_TYPES)}"
            )
        if self.training_steps <= 0:
            raise ValueError("training_steps must be positive")
        if self.eval_games <= 0:
            raise ValueError("eval_games must be positive")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")

    def get_checkpoint_dir(self, results_dir: Path) -> Path:
        """Get the full checkpoint directory path.

        Args:
            results_dir: Base results directory

        Returns:
            Full path to checkpoint directory for this experiment
        """
        if self.checkpoint_dir:
            return results_dir / self.checkpoint_dir
        return results_dir / self.name / "checkpoints"


@dataclass
class OrchestratorConfig:
    """Configuration for the training orchestrator.

    Defines global settings for experiment orchestration.

    Attributes:
        experiments: List of experiment configurations
        parallel_runs: Number of experiments to run in parallel (1 = sequential)
        results_dir: Base directory for all results
        report_format: Output format for comparison reports
    """
    experiments: List[ExperimentConfig]
    parallel_runs: int = 1
    results_dir: str = "results"
    report_format: Literal["markdown", "json", "both"] = "both"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.experiments:
            raise ValueError("At least one experiment must be defined")
        if self.parallel_runs <= 0:
            raise ValueError("parallel_runs must be positive")

        # Check for duplicate experiment names
        names = [exp.name for exp in self.experiments]
        if len(names) != len(set(names)):
            raise ValueError("Experiment names must be unique")

    @property
    def results_path(self) -> Path:
        """Get results directory as Path."""
        return Path(self.results_dir)


def load_config(config_path: str) -> OrchestratorConfig:
    """Load orchestrator configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        OrchestratorConfig with all experiment definitions

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("Config file is empty")

    # Parse experiments
    experiments = []
    raw_experiments = raw.get("experiments", [])

    if not raw_experiments:
        raise ValueError("No experiments defined in config")

    for exp_dict in raw_experiments:
        # Handle hyperparameters field
        hyperparams = exp_dict.pop("hyperparameters", None)

        exp = ExperimentConfig(
            name=exp_dict["name"],
            algorithm=exp_dict["algorithm"],
            representation=exp_dict["representation"],
            reward_type=exp_dict["reward_type"],
            training_steps=exp_dict.get("training_steps", 100000),
            eval_games=exp_dict.get("eval_games", 100),
            n_envs=exp_dict.get("n_envs", 32),
            hyperparameters=hyperparams,
            checkpoint_dir=exp_dict.get("checkpoint_dir"),
        )
        experiments.append(exp)

    # Parse orchestrator settings
    orch_settings = raw.get("orchestrator", {})

    return OrchestratorConfig(
        experiments=experiments,
        parallel_runs=orch_settings.get("parallel_runs", 1),
        results_dir=orch_settings.get("results_dir", "results"),
        report_format=orch_settings.get("report_format", "both"),
    )


def save_config(config: OrchestratorConfig, config_path: str) -> None:
    """Save orchestrator configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to output YAML file
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    experiments = []
    for exp in config.experiments:
        exp_dict = {
            "name": exp.name,
            "algorithm": exp.algorithm,
            "representation": exp.representation,
            "reward_type": exp.reward_type,
            "training_steps": exp.training_steps,
            "eval_games": exp.eval_games,
            "n_envs": exp.n_envs,
        }
        if exp.hyperparameters:
            exp_dict["hyperparameters"] = exp.hyperparameters
        if exp.checkpoint_dir:
            exp_dict["checkpoint_dir"] = exp.checkpoint_dir
        experiments.append(exp_dict)

    output = {
        "orchestrator": {
            "parallel_runs": config.parallel_runs,
            "results_dir": config.results_dir,
            "report_format": config.report_format,
        },
        "experiments": experiments,
    }

    with open(path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


def create_quick_config(
    algorithms: List[str],
    representations: List[str],
    reward_types: List[str],
    training_steps: int = 100000,
    eval_games: int = 100,
    results_dir: str = "results",
) -> OrchestratorConfig:
    """Create a configuration for quick comparison experiments.

    Generates all combinations of the specified parameters.

    Args:
        algorithms: List of algorithm names
        representations: List of representation types
        reward_types: List of reward types
        training_steps: Training steps per experiment
        eval_games: Evaluation games per experiment
        results_dir: Results directory

    Returns:
        OrchestratorConfig with all combinations
    """
    experiments = []
    for algo in algorithms:
        for repr_type in representations:
            for reward in reward_types:
                name = f"{algo}_{repr_type}_{reward}"
                exp = ExperimentConfig(
                    name=name,
                    algorithm=algo,
                    representation=repr_type,
                    reward_type=reward,
                    training_steps=training_steps,
                    eval_games=eval_games,
                )
                experiments.append(exp)

    return OrchestratorConfig(
        experiments=experiments,
        parallel_runs=1,
        results_dir=results_dir,
        report_format="both",
    )
