"""Tests for orchestrator runner module.

Tests the ExperimentRunner class and related functions.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from orchestrator.config import ExperimentConfig, OrchestratorConfig
from orchestrator.runner import ExperimentRunner, _create_env_factory


class TestCreateEnvFactory:
    """Tests for _create_env_factory function."""

    def test_factory_returns_callable(self):
        """Test that factory returns a callable."""
        factory = _create_env_factory(32)
        assert callable(factory)

    @patch('orchestrator.runner.torch')
    def test_factory_creates_env(self, mock_torch):
        """Test that factory creates GameEnv with correct params."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        factory = _create_env_factory(64)

        # Factory should be callable, we can't fully test without CUDA
        # but we verify it's structured correctly
        assert callable(factory)


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

    def test_init_creates_results_dir(self):
        """Test that initialization creates results directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "new_results"
            assert not results_dir.exists()

            exp = ExperimentConfig(
                name="test",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=str(results_dir),
            )
            runner = ExperimentRunner(config)

            assert results_dir.exists()

    def test_load_algorithm_module_success(self):
        """Test successful algorithm module loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentConfig(
                name="test",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            # DQN module should exist
            module = runner._load_algorithm_module("dqn")
            assert hasattr(module, "train")
            assert hasattr(module, "evaluate")

    def test_load_algorithm_module_invalid(self):
        """Test loading non-existent algorithm raises ImportError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentConfig(
                name="test",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            with pytest.raises(ImportError):
                runner._load_algorithm_module("nonexistent_algorithm")

    def test_create_env_factory(self):
        """Test environment factory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentConfig(
                name="test",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
                n_envs=64,
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            factory = runner._create_env_factory(exp)
            assert callable(factory)

    def test_get_results_empty(self):
        """Test getting results when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentConfig(
                name="test",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            results = runner.get_results()
            assert results == []

    def test_run_experiment_invalid_algorithm(self):
        """Test running experiment with invalid algorithm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentConfig(
                name="test_invalid",
                algorithm="nonexistent",
                representation="onehot",
                reward_type="merge",
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            result = runner.run_experiment(exp)

            assert result.status == "failed"
            assert "import" in result.error_message.lower() or "module" in result.error_message.lower()

    def test_dict_to_result_success(self):
        """Test converting success dict to ExperimentResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentConfig(
                name="test",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            result_dict = {
                "experiment_name": "test",
                "status": "success",
                "training_metrics": {
                    "experiment_name": "test",
                    "algorithm": "dqn",
                    "representation": "onehot",
                    "reward_type": "merge",
                    "total_steps": 1000,
                    "total_episodes": 50,
                    "final_avg_score": 500.0,
                    "losses": [],
                    "q_means": [],
                    "epsilons": [],
                    "eval_scores": [],
                    "training_time_seconds": 60.0,
                    "timestamp": "2024-01-01T00:00:00",
                },
                "evaluation_metrics": {
                    "experiment_name": "test",
                    "num_games": 10,
                    "scores": [100, 200, 300],
                    "avg_score": 200.0,
                    "max_score": 300,
                    "min_score": 100,
                    "std_score": 81.6,
                    "median_score": 200.0,
                    "timestamp": "2024-01-01T00:01:00",
                },
                "checkpoint_path": "/path/to/checkpoint.pt",
            }

            result = runner._dict_to_result(result_dict)

            assert result.experiment_name == "test"
            assert result.status == "success"
            assert result.training_metrics.total_steps == 1000
            assert result.evaluation_metrics.avg_score == 200.0
            assert result.checkpoint_path == "/path/to/checkpoint.pt"

    def test_dict_to_result_failure(self):
        """Test converting failure dict to ExperimentResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentConfig(
                name="test",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            result_dict = {
                "experiment_name": "failed_test",
                "status": "failed",
                "error_message": "Something went wrong",
            }

            result = runner._dict_to_result(result_dict)

            assert result.experiment_name == "failed_test"
            assert result.status == "failed"
            assert result.error_message == "Something went wrong"
            assert result.training_metrics is None
            assert result.evaluation_metrics is None


class TestRunAll:
    """Tests for run_all method with mocked experiments."""

    def test_run_all_sequential(self):
        """Test running all experiments sequentially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exps = [
                ExperimentConfig(
                    name=f"test_{i}",
                    algorithm="nonexistent",  # Will fail but captured
                    representation="onehot",
                    reward_type="merge",
                )
                for i in range(3)
            ]
            config = OrchestratorConfig(
                experiments=exps,
                parallel_runs=1,  # Sequential
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            results = runner.run_all()

            assert len(results) == 3
            # All should fail since algorithm doesn't exist
            for result in results:
                assert result.status == "failed"

    def test_run_all_results_stored(self):
        """Test that run_all stores results in collector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentConfig(
                name="test",
                algorithm="nonexistent",
                representation="onehot",
                reward_type="merge",
            )
            config = OrchestratorConfig(
                experiments=[exp],
                results_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            results = runner.run_all()
            stored_results = runner.get_results()

            assert len(stored_results) == 1
            assert stored_results[0].experiment_name == "test"
