"""Tests for orchestrator configuration module.

Tests the ExperimentConfig, OrchestratorConfig, and config I/O functions.
"""

import pytest
import tempfile
from pathlib import Path

from orchestrator.config import (
    ExperimentConfig,
    OrchestratorConfig,
    load_config,
    save_config,
    create_quick_config,
    VALID_REPRESENTATIONS,
    VALID_REWARD_TYPES,
)


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_valid_config_creation(self):
        """Test creating a valid experiment config."""
        config = ExperimentConfig(
            name="test_exp",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        assert config.name == "test_exp"
        assert config.algorithm == "dqn"
        assert config.representation == "onehot"
        assert config.reward_type == "merge"
        # Check defaults
        assert config.training_steps == 100000
        assert config.eval_games == 100
        assert config.n_envs == 32

    def test_custom_training_params(self):
        """Test custom training parameters."""
        config = ExperimentConfig(
            name="custom",
            algorithm="dqn",
            representation="embedding",
            reward_type="spawn",
            training_steps=50000,
            eval_games=50,
            n_envs=64,
        )
        assert config.training_steps == 50000
        assert config.eval_games == 50
        assert config.n_envs == 64

    def test_invalid_representation_raises(self):
        """Test that invalid representation raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ExperimentConfig(
                name="bad",
                algorithm="dqn",
                representation="invalid_repr",
                reward_type="merge",
            )
        assert "Invalid representation" in str(exc_info.value)

    def test_invalid_reward_type_raises(self):
        """Test that invalid reward type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ExperimentConfig(
                name="bad",
                algorithm="dqn",
                representation="onehot",
                reward_type="invalid_reward",
            )
        assert "Invalid reward_type" in str(exc_info.value)

    def test_invalid_training_steps_raises(self):
        """Test that non-positive training steps raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ExperimentConfig(
                name="bad",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
                training_steps=0,
            )
        assert "training_steps must be positive" in str(exc_info.value)

    def test_invalid_eval_games_raises(self):
        """Test that non-positive eval games raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ExperimentConfig(
                name="bad",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
                eval_games=-1,
            )
        assert "eval_games must be positive" in str(exc_info.value)

    def test_get_checkpoint_dir_default(self):
        """Test default checkpoint directory path."""
        config = ExperimentConfig(
            name="test_exp",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        results_dir = Path("/results")
        checkpoint_dir = config.get_checkpoint_dir(results_dir)
        assert checkpoint_dir == Path("/results/test_exp/checkpoints")

    def test_get_checkpoint_dir_custom(self):
        """Test custom checkpoint directory path."""
        config = ExperimentConfig(
            name="test_exp",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            checkpoint_dir="custom/path",
        )
        results_dir = Path("/results")
        checkpoint_dir = config.get_checkpoint_dir(results_dir)
        assert checkpoint_dir == Path("/results/custom/path")

    def test_all_valid_representations(self):
        """Test that all valid representations are accepted."""
        for repr_type in VALID_REPRESENTATIONS:
            config = ExperimentConfig(
                name=f"test_{repr_type}",
                algorithm="dqn",
                representation=repr_type,
                reward_type="merge",
            )
            assert config.representation == repr_type

    def test_all_valid_reward_types(self):
        """Test that all valid reward types are accepted."""
        for reward in VALID_REWARD_TYPES:
            config = ExperimentConfig(
                name=f"test_{reward}",
                algorithm="dqn",
                representation="onehot",
                reward_type=reward,
            )
            assert config.reward_type == reward


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_valid_config_creation(self):
        """Test creating a valid orchestrator config."""
        exp = ExperimentConfig(
            name="exp1",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        config = OrchestratorConfig(experiments=[exp])
        assert len(config.experiments) == 1
        assert config.parallel_runs == 1
        assert config.results_dir == "results"
        assert config.report_format == "both"

    def test_multiple_experiments(self):
        """Test config with multiple experiments."""
        exps = [
            ExperimentConfig(
                name=f"exp{i}",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
            )
            for i in range(3)
        ]
        config = OrchestratorConfig(experiments=exps, parallel_runs=2)
        assert len(config.experiments) == 3
        assert config.parallel_runs == 2

    def test_empty_experiments_raises(self):
        """Test that empty experiments list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OrchestratorConfig(experiments=[])
        assert "At least one experiment" in str(exc_info.value)

    def test_duplicate_names_raises(self):
        """Test that duplicate experiment names raises ValueError."""
        exp1 = ExperimentConfig(
            name="same_name",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        exp2 = ExperimentConfig(
            name="same_name",  # Duplicate
            algorithm="dqn",
            representation="embedding",
            reward_type="spawn",
        )
        with pytest.raises(ValueError) as exc_info:
            OrchestratorConfig(experiments=[exp1, exp2])
        assert "unique" in str(exc_info.value)

    def test_invalid_parallel_runs_raises(self):
        """Test that non-positive parallel_runs raises ValueError."""
        exp = ExperimentConfig(
            name="exp1",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        with pytest.raises(ValueError) as exc_info:
            OrchestratorConfig(experiments=[exp], parallel_runs=0)
        assert "parallel_runs must be positive" in str(exc_info.value)

    def test_results_path_property(self):
        """Test results_path property returns Path."""
        exp = ExperimentConfig(
            name="exp1",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        config = OrchestratorConfig(experiments=[exp], results_dir="/custom/results")
        assert config.results_path == Path("/custom/results")


class TestConfigIO:
    """Tests for config load/save functions."""

    def test_save_and_load_config(self):
        """Test round-trip save and load of config."""
        exp = ExperimentConfig(
            name="test_exp",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            training_steps=50000,
            eval_games=50,
        )
        original_config = OrchestratorConfig(
            experiments=[exp],
            parallel_runs=2,
            results_dir="custom_results",
            report_format="json",
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            save_config(original_config, config_path)
            loaded_config = load_config(config_path)

            assert len(loaded_config.experiments) == 1
            assert loaded_config.experiments[0].name == "test_exp"
            assert loaded_config.experiments[0].training_steps == 50000
            assert loaded_config.parallel_runs == 2
            assert loaded_config.results_dir == "custom_results"
            assert loaded_config.report_format == "json"
        finally:
            Path(config_path).unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_empty_config_raises(self):
        """Test loading empty config file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            config_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(config_path)
            assert "empty" in str(exc_info.value)
        finally:
            Path(config_path).unlink()

    def test_load_config_no_experiments_raises(self):
        """Test loading config without experiments raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("orchestrator:\n  parallel_runs: 1\n")
            config_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(config_path)
            assert "No experiments" in str(exc_info.value)
        finally:
            Path(config_path).unlink()

    def test_save_config_with_hyperparameters(self):
        """Test saving config with hyperparameters override."""
        exp = ExperimentConfig(
            name="test_exp",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            hyperparameters={"learning_rate": 0.001, "batch_size": 64},
        )
        config = OrchestratorConfig(experiments=[exp])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            save_config(config, config_path)
            loaded = load_config(config_path)
            assert loaded.experiments[0].hyperparameters == {"learning_rate": 0.001, "batch_size": 64}
        finally:
            Path(config_path).unlink()


class TestCreateQuickConfig:
    """Tests for create_quick_config helper."""

    def test_single_combination(self):
        """Test creating config with single algorithm/repr/reward."""
        config = create_quick_config(
            algorithms=["dqn"],
            representations=["onehot"],
            reward_types=["merge"],
        )
        assert len(config.experiments) == 1
        exp = config.experiments[0]
        assert exp.name == "dqn_onehot_merge"
        assert exp.algorithm == "dqn"
        assert exp.representation == "onehot"
        assert exp.reward_type == "merge"

    def test_multiple_combinations(self):
        """Test creating config with multiple combinations."""
        config = create_quick_config(
            algorithms=["dqn", "reinforce"],
            representations=["onehot", "embedding"],
            reward_types=["merge", "spawn"],
        )
        # 2 * 2 * 2 = 8 combinations
        assert len(config.experiments) == 8

        # Verify all expected names exist
        expected_names = [
            "dqn_onehot_merge", "dqn_onehot_spawn",
            "dqn_embedding_merge", "dqn_embedding_spawn",
            "reinforce_onehot_merge", "reinforce_onehot_spawn",
            "reinforce_embedding_merge", "reinforce_embedding_spawn",
        ]
        actual_names = [exp.name for exp in config.experiments]
        assert set(actual_names) == set(expected_names)

    def test_custom_training_params(self):
        """Test custom training parameters in quick config."""
        config = create_quick_config(
            algorithms=["dqn"],
            representations=["onehot"],
            reward_types=["merge"],
            training_steps=50000,
            eval_games=200,
            results_dir="custom_dir",
        )
        exp = config.experiments[0]
        assert exp.training_steps == 50000
        assert exp.eval_games == 200
        assert config.results_dir == "custom_dir"
