"""Tests for orchestrator report module.

Tests the ReportGenerator class.
"""

import json
import pytest
import tempfile
from pathlib import Path

from orchestrator.metrics import (
    EvaluationMetrics,
    ExperimentResult,
    MetricsCollector,
    TrainingMetrics,
)
from orchestrator.report import ReportGenerator


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def _create_test_results(self, results_dir: str):
        """Helper to create test results in a directory."""
        collector = MetricsCollector(results_dir)

        # Successful experiment 1 (best)
        collector.add_result(ExperimentResult(
            experiment_name="exp_best",
            status="success",
            training_metrics=TrainingMetrics(
                experiment_name="exp_best",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
                total_steps=100000,
                total_episodes=5000,
                final_avg_score=2000.0,
                training_time_seconds=3600.0,
            ),
            evaluation_metrics=EvaluationMetrics.from_scores(
                "exp_best", [2000, 2200, 1800]
            ),
            checkpoint_path="/results/exp_best/checkpoint.pt",
        ))

        # Successful experiment 2 (second)
        collector.add_result(ExperimentResult(
            experiment_name="exp_second",
            status="success",
            training_metrics=TrainingMetrics(
                experiment_name="exp_second",
                algorithm="dqn",
                representation="embedding",
                reward_type="spawn",
                total_steps=100000,
                total_episodes=4000,
                final_avg_score=1500.0,
                training_time_seconds=3000.0,
            ),
            evaluation_metrics=EvaluationMetrics.from_scores(
                "exp_second", [1500, 1600, 1400]
            ),
            checkpoint_path="/results/exp_second/checkpoint.pt",
        ))

        # Failed experiment
        collector.add_result(ExperimentResult(
            experiment_name="exp_failed",
            status="failed",
            error_message="Module import error",
        ))

        return collector

    def test_init_loads_results(self):
        """Test that initialization loads existing results.

        Note: load_existing_results() only loads experiments with metrics files,
        so failed experiments (which have no metrics) are not loaded.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)

            generator = ReportGenerator(tmpdir)
            # Only successful experiments with metrics files are loaded
            # Failed experiments don't have metrics files and can't be recovered
            assert len(generator.metrics_collector.results) == 2

    def test_rank_experiments(self):
        """Test experiment ranking by score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            results = list(generator.metrics_collector.results.values())
            ranked = generator._rank_experiments(results)

            assert len(ranked) == 2  # Only successful experiments
            assert ranked[0]["name"] == "exp_best"  # 2000 avg
            assert ranked[0]["rank"] == 1
            assert ranked[1]["name"] == "exp_second"  # 1500 avg
            assert ranked[1]["rank"] == 2

    def test_rank_experiments_empty(self):
        """Test ranking with no successful experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(tmpdir)
            collector.add_result(ExperimentResult(
                experiment_name="failed",
                status="failed",
                error_message="Error",
            ))

            generator = ReportGenerator(tmpdir)
            results = list(generator.metrics_collector.results.values())
            ranked = generator._rank_experiments(results)

            assert len(ranked) == 0


class TestGenerateMarkdown:
    """Tests for markdown report generation."""

    def _create_test_results(self, results_dir: str):
        """Helper to create test results."""
        collector = MetricsCollector(results_dir)

        collector.add_result(ExperimentResult(
            experiment_name="exp1",
            status="success",
            training_metrics=TrainingMetrics(
                experiment_name="exp1",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
                total_steps=50000,
                total_episodes=2500,
                training_time_seconds=1800.0,
            ),
            evaluation_metrics=EvaluationMetrics.from_scores("exp1", [1000, 1100]),
            checkpoint_path="/path/exp1.pt",
        ))

        collector.add_result(ExperimentResult(
            experiment_name="exp_failed",
            status="failed",
            error_message="Test error message",
        ))

        return collector

    def test_markdown_contains_header(self):
        """Test that markdown contains proper header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            md = generator.generate_markdown()
            assert "# Training Orchestrator Report" in md
            assert "Generated:" in md

    def test_markdown_contains_summary(self):
        """Test that markdown contains summary section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = self._create_test_results(tmpdir)
            # Create generator with pre-populated collector to include failed exp
            generator = ReportGenerator(tmpdir)
            generator.metrics_collector = collector  # Use the in-memory collector

            md = generator.generate_markdown()
            assert "## Summary" in md
            assert "**Total Experiments:** 2" in md
            assert "**Successful:** 1" in md
            assert "**Failed:** 1" in md

    def test_markdown_contains_rankings_table(self):
        """Test that markdown contains rankings table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            md = generator.generate_markdown()
            assert "## Experiment Rankings" in md
            assert "| Rank |" in md
            assert "| 1 |" in md  # Rank 1

    def test_markdown_contains_top_performer(self):
        """Test that markdown highlights top performer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            md = generator.generate_markdown()
            assert "### Top Performer" in md
            assert "**exp1**" in md

    def test_markdown_contains_experiment_details(self):
        """Test that markdown contains per-experiment details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            md = generator.generate_markdown()
            assert "## Experiment Details" in md
            assert "### exp1" in md
            assert "**Training:**" in md
            assert "**Evaluation:**" in md

    def test_markdown_contains_failed_experiments(self):
        """Test that markdown lists failed experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = self._create_test_results(tmpdir)
            # Create generator with pre-populated collector to include failed exp
            generator = ReportGenerator(tmpdir)
            generator.metrics_collector = collector  # Use the in-memory collector

            md = generator.generate_markdown()
            assert "## Failed Experiments" in md
            assert "exp_failed" in md
            assert "Test error message" in md


class TestGenerateJSON:
    """Tests for JSON report generation."""

    def _create_test_results(self, results_dir: str):
        """Helper to create test results."""
        collector = MetricsCollector(results_dir)

        collector.add_result(ExperimentResult(
            experiment_name="exp1",
            status="success",
            training_metrics=TrainingMetrics(
                experiment_name="exp1",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
                total_steps=50000,
                total_episodes=2500,
                training_time_seconds=1800.0,
            ),
            evaluation_metrics=EvaluationMetrics.from_scores("exp1", [1000, 1100]),
            checkpoint_path="/path/exp1.pt",
        ))

        return collector

    def test_json_structure(self):
        """Test JSON report structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            report = generator.generate_json()

            assert "generated_at" in report
            assert "results_dir" in report
            assert "summary" in report
            assert "rankings" in report
            assert "experiments" in report

    def test_json_summary(self):
        """Test JSON summary section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            report = generator.generate_json()
            summary = report["summary"]

            assert summary["total_experiments"] == 1
            assert summary["successful"] == 1
            assert summary["failed"] == 0

    def test_json_rankings(self):
        """Test JSON rankings section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            report = generator.generate_json()
            rankings = report["rankings"]

            assert len(rankings) == 1
            assert rankings[0]["rank"] == 1
            assert rankings[0]["name"] == "exp1"
            assert rankings[0]["avg_score"] == 1050.0

    def test_json_experiments(self):
        """Test JSON experiments section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            report = generator.generate_json()
            experiments = report["experiments"]

            assert "exp1" in experiments
            exp1 = experiments["exp1"]
            assert exp1["status"] == "success"
            assert "training" in exp1
            assert "evaluation" in exp1
            assert exp1["training"]["algorithm"] == "dqn"
            assert exp1["evaluation"]["avg_score"] == 1050.0


class TestSaveReports:
    """Tests for save_reports method."""

    def _create_test_results(self, results_dir: str):
        """Helper to create test results."""
        collector = MetricsCollector(results_dir)
        collector.add_result(ExperimentResult(
            experiment_name="test",
            status="success",
            evaluation_metrics=EvaluationMetrics.from_scores("test", [500]),
        ))
        return collector

    def test_save_markdown_only(self):
        """Test saving markdown report only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            saved = generator.save_reports(formats=["markdown"])

            assert "markdown" in saved
            assert Path(saved["markdown"]).exists()
            assert "json" not in saved

    def test_save_json_only(self):
        """Test saving JSON report only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            saved = generator.save_reports(formats=["json"])

            assert "json" in saved
            assert Path(saved["json"]).exists()
            assert "markdown" not in saved

    def test_save_both_formats(self):
        """Test saving both report formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            saved = generator.save_reports(formats=["both"])

            assert "markdown" in saved
            assert "json" in saved
            assert Path(saved["markdown"]).exists()
            assert Path(saved["json"]).exists()

    def test_save_to_custom_directory(self):
        """Test saving reports to custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            output_dir = Path(tmpdir) / "custom_output"
            saved = generator.save_reports(
                output_dir=str(output_dir),
                formats=["markdown", "json"],
            )

            assert output_dir.exists()
            assert (output_dir / "report.md").exists()
            assert (output_dir / "report.json").exists()

    def test_saved_markdown_is_valid(self):
        """Test that saved markdown file is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            saved = generator.save_reports(formats=["markdown"])

            with open(saved["markdown"]) as f:
                content = f.read()
                assert "# Training Orchestrator Report" in content

    def test_saved_json_is_valid(self):
        """Test that saved JSON file is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_results(tmpdir)
            generator = ReportGenerator(tmpdir)

            saved = generator.save_reports(formats=["json"])

            with open(saved["json"]) as f:
                data = json.load(f)
                assert "summary" in data
                assert "rankings" in data
