"""
Tests for Sweep Runner.

Tests the main sweep orchestration functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

from sweep.runner import SweepRunner
from sweep.study_factory import ALGORITHMS


class TestSweepRunner:
    """Tests for SweepRunner class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_runner_creation(self, temp_dir):
        """SweepRunner can be created."""
        runner = SweepRunner(
            n_trials=5,
            storage_path=f"sqlite:///{temp_dir}/test.db",
            algorithms=["dqn"],
            representations=["onehot"],
            reward_types=["merge"],
        )
        assert runner is not None

    def test_runner_total_studies(self, temp_dir):
        """Runner calculates correct total studies."""
        runner = SweepRunner(
            n_trials=5,
            storage_path=f"sqlite:///{temp_dir}/test.db",
            algorithms=["dqn", "double_dqn"],
            representations=["onehot", "embedding"],
            reward_types=["merge", "spawn"],
        )
        assert runner.factory.total_studies == 8  # 2x2x2

    def test_runner_default_all_algorithms(self, temp_dir):
        """Runner defaults to all 20 algorithms."""
        runner = SweepRunner(
            n_trials=5,
            storage_path=f"sqlite:///{temp_dir}/test.db",
        )
        assert len(runner.factory.algorithms) == 20
        assert runner.factory.total_studies == 200

    def test_get_completed_studies_empty(self, temp_dir):
        """get_completed_studies returns empty set when no results."""
        runner = SweepRunner(
            n_trials=5,
            storage_path=f"sqlite:///{temp_dir}/test.db",
            status_file=f"{temp_dir}/status.jsonl",
            results_file=f"{temp_dir}/results.jsonl",
        )
        completed = runner.get_completed_studies()
        assert len(completed) == 0

    def test_get_completed_studies_with_results(self, temp_dir):
        """get_completed_studies returns studies from results file."""
        results_file = Path(temp_dir) / "results.jsonl"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Write some completed studies
        with open(results_file, "w") as f:
            f.write(json.dumps({"study": "study1", "status": "completed"}) + "\n")
            f.write(json.dumps({"study": "study2", "status": "completed"}) + "\n")
            f.write(json.dumps({"study": "study3", "status": "failed"}) + "\n")

        runner = SweepRunner(
            n_trials=5,
            storage_path=f"sqlite:///{temp_dir}/test.db",
            status_file=f"{temp_dir}/status.jsonl",
            results_file=str(results_file),
        )

        completed = runner.get_completed_studies()
        assert len(completed) == 2
        assert "study1" in completed
        assert "study2" in completed
        assert "study3" not in completed  # Failed, not completed

    def test_runner_creates_storage_directory(self, temp_dir):
        """Runner creates storage directory if needed."""
        nested_path = f"{temp_dir}/nested/dir/test.db"
        runner = SweepRunner(
            n_trials=5,
            storage_path=f"sqlite:///{nested_path}",
        )
        assert Path(nested_path).parent.exists()

    def test_status_method(self, temp_dir):
        """status() method runs without error."""
        runner = SweepRunner(
            n_trials=5,
            storage_path=f"sqlite:///{temp_dir}/test.db",
            status_file=f"{temp_dir}/status.jsonl",
            results_file=f"{temp_dir}/results.jsonl",
        )
        # Should not raise
        runner.status()


class TestSweepRunnerParameters:
    """Tests for SweepRunner parameter handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_n_trials_parameter(self, temp_dir):
        """n_trials parameter is passed to factory."""
        runner = SweepRunner(
            n_trials=10,
            storage_path=f"sqlite:///{temp_dir}/test.db",
        )
        assert runner.factory.n_trials == 10

    def test_epochs_parameter(self, temp_dir):
        """epochs_per_trial parameter is passed to factory."""
        runner = SweepRunner(
            epochs_per_trial=100,
            storage_path=f"sqlite:///{temp_dir}/test.db",
        )
        assert runner.factory.epochs_per_trial == 100

    def test_steps_parameter(self, temp_dir):
        """steps_per_epoch parameter is passed to factory."""
        runner = SweepRunner(
            steps_per_epoch=500,
            storage_path=f"sqlite:///{temp_dir}/test.db",
        )
        assert runner.factory.steps_per_epoch == 500

    def test_eval_games_parameter(self, temp_dir):
        """eval_games_per_epoch parameter is passed to factory."""
        runner = SweepRunner(
            eval_games_per_epoch=25,
            storage_path=f"sqlite:///{temp_dir}/test.db",
        )
        assert runner.factory.eval_games_per_epoch == 25


class TestSweepRunnerFiltering:
    """Tests for algorithm/representation/reward filtering."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_single_algorithm(self, temp_dir):
        """Can filter to single algorithm."""
        runner = SweepRunner(
            storage_path=f"sqlite:///{temp_dir}/test.db",
            algorithms=["dqn"],
        )
        assert runner.factory.algorithms == ["dqn"]
        assert runner.factory.total_studies == 10  # 1 algo x 5 repr x 2 rewards

    def test_multiple_algorithms(self, temp_dir):
        """Can filter to multiple algorithms."""
        runner = SweepRunner(
            storage_path=f"sqlite:///{temp_dir}/test.db",
            algorithms=["dqn", "a2c", "ppo_gae"],
        )
        assert len(runner.factory.algorithms) == 3
        assert runner.factory.total_studies == 30  # 3 algo x 5 repr x 2 rewards

    def test_single_representation(self, temp_dir):
        """Can filter to single representation."""
        runner = SweepRunner(
            storage_path=f"sqlite:///{temp_dir}/test.db",
            representations=["onehot"],
        )
        assert runner.factory.representations == ["onehot"]
        assert runner.factory.total_studies == 40  # 20 algo x 1 repr x 2 rewards

    def test_single_reward(self, temp_dir):
        """Can filter to single reward type."""
        runner = SweepRunner(
            storage_path=f"sqlite:///{temp_dir}/test.db",
            reward_types=["merge"],
        )
        assert runner.factory.reward_types == ["merge"]
        assert runner.factory.total_studies == 100  # 20 algo x 5 repr x 1 reward

    def test_combined_filtering(self, temp_dir):
        """Can combine multiple filters."""
        runner = SweepRunner(
            storage_path=f"sqlite:///{temp_dir}/test.db",
            algorithms=["dqn"],
            representations=["onehot"],
            reward_types=["merge"],
        )
        assert runner.factory.total_studies == 1
