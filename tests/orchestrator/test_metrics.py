"""Tests for orchestrator metrics module.

Tests the TrainingMetrics, EvaluationMetrics, ExperimentResult,
and MetricsCollector classes.
"""

import json
import pytest
import tempfile
from pathlib import Path

from orchestrator.metrics import (
    TrainingMetrics,
    EvaluationMetrics,
    ExperimentResult,
    MetricsCollector,
)


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_creation_with_defaults(self):
        """Test creating TrainingMetrics with minimal params."""
        metrics = TrainingMetrics(
            experiment_name="test",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        assert metrics.experiment_name == "test"
        assert metrics.algorithm == "dqn"
        assert metrics.total_steps == 0
        assert metrics.total_episodes == 0
        assert metrics.final_avg_score == 0.0
        assert metrics.losses == []
        assert metrics.q_means == []
        assert metrics.training_time_seconds == 0.0
        assert metrics.timestamp is not None

    def test_creation_with_all_fields(self):
        """Test creating TrainingMetrics with all fields."""
        metrics = TrainingMetrics(
            experiment_name="test",
            algorithm="dqn",
            representation="embedding",
            reward_type="spawn",
            total_steps=100000,
            total_episodes=5000,
            final_avg_score=1234.5,
            losses=[0.5, 0.4, 0.3],
            q_means=[10.0, 11.0, 12.0],
            epsilons=[1.0, 0.5, 0.1],
            eval_scores=[{"step": 1000, "avg_score": 500}],
            training_time_seconds=3600.0,
        )
        assert metrics.total_steps == 100000
        assert metrics.total_episodes == 5000
        assert metrics.final_avg_score == 1234.5
        assert len(metrics.losses) == 3
        assert metrics.training_time_seconds == 3600.0


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_from_scores_empty(self):
        """Test creating from empty scores list."""
        metrics = EvaluationMetrics.from_scores("test", [])
        assert metrics.experiment_name == "test"
        assert metrics.num_games == 0
        assert metrics.scores == []
        assert metrics.avg_score == 0.0
        assert metrics.max_score == 0
        assert metrics.min_score == 0
        assert metrics.std_score == 0.0
        assert metrics.median_score == 0.0

    def test_from_scores_single(self):
        """Test creating from single score."""
        metrics = EvaluationMetrics.from_scores("test", [1000])
        assert metrics.num_games == 1
        assert metrics.avg_score == 1000.0
        assert metrics.max_score == 1000
        assert metrics.min_score == 1000
        assert metrics.std_score == 0.0  # No std for single value
        assert metrics.median_score == 1000.0

    def test_from_scores_multiple(self):
        """Test creating from multiple scores."""
        scores = [100, 200, 300, 400, 500]
        metrics = EvaluationMetrics.from_scores("test", scores)
        assert metrics.num_games == 5
        assert metrics.avg_score == 300.0
        assert metrics.max_score == 500
        assert metrics.min_score == 100
        assert metrics.median_score == 300.0
        assert metrics.std_score > 0

    def test_from_scores_statistics(self):
        """Test statistical calculations."""
        # Known values for verification
        scores = [10, 20, 30, 40, 50]
        metrics = EvaluationMetrics.from_scores("test", scores)

        # Mean = (10+20+30+40+50)/5 = 30
        assert metrics.avg_score == 30.0

        # Median of [10, 20, 30, 40, 50] = 30
        assert metrics.median_score == 30.0

        # Std dev should be sqrt(200) = 14.14... (sample std dev)
        import statistics
        expected_std = statistics.stdev(scores)
        assert abs(metrics.std_score - expected_std) < 0.01


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_successful_result(self):
        """Test creating successful experiment result."""
        training = TrainingMetrics(
            experiment_name="test",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        evaluation = EvaluationMetrics.from_scores("test", [1000, 2000])

        result = ExperimentResult(
            experiment_name="test",
            status="success",
            training_metrics=training,
            evaluation_metrics=evaluation,
            checkpoint_path="/path/to/checkpoint.pt",
        )
        assert result.status == "success"
        assert result.training_metrics is not None
        assert result.evaluation_metrics is not None
        assert result.error_message is None

    def test_failed_result(self):
        """Test creating failed experiment result."""
        result = ExperimentResult(
            experiment_name="failed_test",
            status="failed",
            error_message="Import error: module not found",
        )
        assert result.status == "failed"
        assert result.error_message == "Import error: module not found"
        assert result.training_metrics is None
        assert result.evaluation_metrics is None


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_add_and_get_result(self):
        """Test adding and retrieving results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(tmpdir)

            result = ExperimentResult(
                experiment_name="test",
                status="success",
                training_metrics=TrainingMetrics(
                    experiment_name="test",
                    algorithm="dqn",
                    representation="onehot",
                    reward_type="merge",
                ),
                evaluation_metrics=EvaluationMetrics.from_scores("test", [1000]),
            )
            collector.add_result(result)

            retrieved = collector.get_result("test")
            assert retrieved is not None
            assert retrieved.experiment_name == "test"
            assert retrieved.status == "success"

    def test_get_nonexistent_result(self):
        """Test getting non-existent result returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(tmpdir)
            assert collector.get_result("nonexistent") is None

    def test_get_successful_results(self):
        """Test filtering successful results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(tmpdir)

            # Add successful result
            collector.add_result(ExperimentResult(
                experiment_name="success1",
                status="success",
                evaluation_metrics=EvaluationMetrics.from_scores("success1", [1000]),
            ))

            # Add failed result
            collector.add_result(ExperimentResult(
                experiment_name="failed1",
                status="failed",
                error_message="Error",
            ))

            successful = collector.get_successful_results()
            assert len(successful) == 1
            assert successful[0].experiment_name == "success1"

    def test_get_failed_results(self):
        """Test filtering failed results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(tmpdir)

            collector.add_result(ExperimentResult(
                experiment_name="success1",
                status="success",
                evaluation_metrics=EvaluationMetrics.from_scores("success1", [1000]),
            ))

            collector.add_result(ExperimentResult(
                experiment_name="failed1",
                status="failed",
                error_message="Error 1",
            ))

            collector.add_result(ExperimentResult(
                experiment_name="failed2",
                status="failed",
                error_message="Error 2",
            ))

            failed = collector.get_failed_results()
            assert len(failed) == 2

    def test_result_saved_to_files(self):
        """Test that results are saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(tmpdir)

            result = ExperimentResult(
                experiment_name="test_save",
                status="success",
                training_metrics=TrainingMetrics(
                    experiment_name="test_save",
                    algorithm="dqn",
                    representation="onehot",
                    reward_type="merge",
                    total_steps=1000,
                ),
                evaluation_metrics=EvaluationMetrics.from_scores("test_save", [500, 1000]),
                checkpoint_path="/path/checkpoint.pt",
            )
            collector.add_result(result)

            # Check experiment directory exists
            exp_dir = Path(tmpdir) / "test_save"
            assert exp_dir.exists()

            # Check training metrics file
            training_file = exp_dir / "training_metrics.json"
            assert training_file.exists()
            with open(training_file) as f:
                training_data = json.load(f)
                assert training_data["experiment_name"] == "test_save"
                assert training_data["total_steps"] == 1000

            # Check evaluation metrics file
            eval_file = exp_dir / "evaluation_metrics.json"
            assert eval_file.exists()
            with open(eval_file) as f:
                eval_data = json.load(f)
                assert eval_data["avg_score"] == 750.0

            # Check all_results.jsonl
            results_file = Path(tmpdir) / "all_results.jsonl"
            assert results_file.exists()
            with open(results_file) as f:
                line = f.readline()
                record = json.loads(line)
                assert record["experiment_name"] == "test_save"
                assert record["status"] == "success"

    def test_get_comparison_data(self):
        """Test comparison data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(tmpdir)

            # Add experiments with different scores
            collector.add_result(ExperimentResult(
                experiment_name="exp1",
                status="success",
                evaluation_metrics=EvaluationMetrics.from_scores("exp1", [1000, 1200]),
                checkpoint_path="/path/exp1.pt",
            ))
            collector.add_result(ExperimentResult(
                experiment_name="exp2",
                status="success",
                evaluation_metrics=EvaluationMetrics.from_scores("exp2", [500, 600]),
                checkpoint_path="/path/exp2.pt",
            ))
            collector.add_result(ExperimentResult(
                experiment_name="exp3",
                status="failed",
                error_message="Error",
            ))

            data = collector.get_comparison_data()
            assert data["total_experiments"] == 3
            assert data["successful"] == 2
            assert data["failed"] == 1

            # Experiments should be sorted by avg_score (descending)
            assert len(data["experiments"]) == 2
            assert data["experiments"][0]["name"] == "exp1"  # 1100 avg
            assert data["experiments"][1]["name"] == "exp2"  # 550 avg

    def test_load_existing_results(self):
        """Test loading results from existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First, create results with one collector
            collector1 = MetricsCollector(tmpdir)
            collector1.add_result(ExperimentResult(
                experiment_name="existing",
                status="success",
                training_metrics=TrainingMetrics(
                    experiment_name="existing",
                    algorithm="dqn",
                    representation="onehot",
                    reward_type="merge",
                ),
                evaluation_metrics=EvaluationMetrics.from_scores("existing", [1000]),
            ))

            # Create new collector and load
            collector2 = MetricsCollector(tmpdir)
            collector2.load_existing_results()

            assert "existing" in collector2.results
            result = collector2.get_result("existing")
            assert result.training_metrics.algorithm == "dqn"
            assert result.evaluation_metrics.avg_score == 1000.0

    def test_summary(self):
        """Test summary string generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(tmpdir)

            collector.add_result(ExperimentResult(
                experiment_name="top_exp",
                status="success",
                evaluation_metrics=EvaluationMetrics.from_scores("top_exp", [2000]),
            ))
            collector.add_result(ExperimentResult(
                experiment_name="bottom_exp",
                status="success",
                evaluation_metrics=EvaluationMetrics.from_scores("bottom_exp", [500]),
            ))

            summary = collector.summary()
            assert "Metrics Summary" in summary
            assert "Total experiments: 2" in summary
            assert "Successful: 2" in summary
            assert "top_exp" in summary
