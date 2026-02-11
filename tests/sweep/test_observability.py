"""
Tests for Observability Module.

Tests the logging, status tracking, and monitoring capabilities.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from sweep.observability import SweepObserver, StudyMetrics, TrialMetrics


class TestSweepObserver:
    """Tests for SweepObserver class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def observer(self, temp_dir):
        """Create observer with temp files."""
        return SweepObserver(
            status_file=f"{temp_dir}/status.jsonl",
            results_file=f"{temp_dir}/results.jsonl",
            summary_file=f"{temp_dir}/summary.json",
        )

    def test_sweep_started_logs_event(self, observer):
        """sweep_started logs to status file."""
        observer.sweep_started(total_studies=200, algorithms=["dqn", "a2c"])

        with open(observer.status_file, "r") as f:
            events = [json.loads(line) for line in f]

        assert len(events) == 1
        assert events[0]["event"] == "sweep_started"
        assert events[0]["total_studies"] == 200
        assert "ts" in events[0]

    def test_sweep_started_sets_state(self, observer):
        """sweep_started sets observer state correctly."""
        observer.sweep_started(total_studies=200, algorithms=["dqn"])

        assert observer.total_studies == 200
        assert observer.completed_studies == 0
        assert observer.failed_studies == 0
        assert observer.sweep_start_time is not None

    def test_study_started_logs_event(self, observer):
        """study_started logs to status file."""
        observer.sweep_started(total_studies=1, algorithms=["dqn"])
        observer.study_started(study_name="dqn_onehot_merge", index=1, n_trials=50)

        with open(observer.status_file, "r") as f:
            events = [json.loads(line) for line in f]

        assert len(events) == 2
        assert events[1]["event"] == "study_started"
        assert events[1]["study"] == "dqn_onehot_merge"
        assert events[1]["index"] == 1

    def test_trial_completed_logs_event(self, observer):
        """trial_completed logs to status file."""
        observer.sweep_started(total_studies=1, algorithms=["dqn"])
        observer.study_started("dqn_onehot_merge", 1, 50)
        observer.trial_completed(
            trial_number=1,
            score=1500.0,
            duration_seconds=30.0,
            params={"lr": 0.001},
        )

        with open(observer.status_file, "r") as f:
            events = [json.loads(line) for line in f]

        trial_events = [e for e in events if e["event"] == "trial_completed"]
        assert len(trial_events) == 1
        assert trial_events[0]["score"] == 1500.0
        assert trial_events[0]["duration_seconds"] == 30.0

    def test_trial_completed_tracks_durations(self, observer):
        """trial_completed tracks durations in rolling window."""
        observer.sweep_started(total_studies=1, algorithms=["dqn"])
        observer.study_started("test", 1, 50)

        for i in range(5):
            observer.trial_completed(i, 1000.0, 10.0 + i, {})

        assert len(observer.trial_durations) == 5
        assert sum(observer.trial_durations) == 60.0

    def test_study_completed_logs_event(self, observer):
        """study_completed logs to status file."""
        observer.sweep_started(total_studies=1, algorithms=["dqn"])
        observer.study_started("dqn_onehot_merge", 1, 50)

        metrics = StudyMetrics(
            study_name="dqn_onehot_merge",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            total_trials=50,
            completed_trials=45,
            pruned_trials=5,
            failed_trials=0,
            best_score=2000.0,
            best_params={"lr": 0.001},
            total_duration_seconds=3600.0,
            avg_trial_duration_seconds=80.0,
            status="completed",
        )

        observer.study_completed("dqn_onehot_merge", metrics)

        with open(observer.status_file, "r") as f:
            events = [json.loads(line) for line in f]

        completed_events = [e for e in events if e["event"] == "study_completed"]
        assert len(completed_events) == 1
        assert completed_events[0]["best_score"] == 2000.0

    def test_study_completed_logs_result(self, observer):
        """study_completed logs to results file."""
        observer.sweep_started(total_studies=1, algorithms=["dqn"])
        observer.study_started("dqn_onehot_merge", 1, 50)

        metrics = StudyMetrics(
            study_name="dqn_onehot_merge",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            total_trials=50,
            completed_trials=45,
            pruned_trials=5,
            failed_trials=0,
            best_score=2000.0,
            best_params={"lr": 0.001},
            total_duration_seconds=3600.0,
            avg_trial_duration_seconds=80.0,
            status="completed",
        )

        observer.study_completed("dqn_onehot_merge", metrics)

        with open(observer.results_file, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 1
        assert results[0]["study"] == "dqn_onehot_merge"
        assert results[0]["status"] == "completed"
        assert results[0]["best_score"] == 2000.0

    def test_study_failed_logs_error(self, observer):
        """study_failed logs error to status and results."""
        observer.sweep_started(total_studies=1, algorithms=["dqn"])
        observer.study_started("dqn_onehot_merge", 1, 50)
        observer.study_failed("dqn_onehot_merge", "Test error message")

        assert observer.failed_studies == 1
        assert len(observer.errors) == 1

        with open(observer.results_file, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert "Test error" in results[0]["error"]

    def test_get_status(self, observer):
        """get_status returns current sweep state."""
        observer.sweep_started(total_studies=10, algorithms=["dqn"])
        observer.completed_studies = 5
        observer.failed_studies = 1

        status = observer.get_status()

        assert status["total_studies"] == 10
        assert status["completed_studies"] == 5
        assert status["failed_studies"] == 1
        assert status["remaining_studies"] == 4
        assert status["progress_percent"] == 60.0

    def test_summary_file_updated(self, observer):
        """Summary file is updated after events."""
        observer.sweep_started(total_studies=1, algorithms=["dqn"])
        observer.study_started("dqn_onehot_merge", 1, 50)

        metrics = StudyMetrics(
            study_name="dqn_onehot_merge",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            total_trials=50,
            completed_trials=50,
            pruned_trials=0,
            failed_trials=0,
            best_score=2000.0,
            best_params={},
            total_duration_seconds=100.0,
            avg_trial_duration_seconds=2.0,
            status="completed",
        )

        observer.study_completed("dqn_onehot_merge", metrics)

        with open(observer.summary_file, "r") as f:
            summary = json.load(f)

        assert summary["completed_studies"] == 1
        assert summary["progress_percent"] == 100.0

    def test_slow_trial_alert(self, observer):
        """Slow trial generates alert event."""
        observer.slow_trial_threshold = 10.0  # 10 seconds
        observer.sweep_started(total_studies=1, algorithms=["dqn"])
        observer.study_started("test", 1, 50)
        observer.trial_completed(1, 1000.0, 20.0, {})  # 20s > 10s threshold

        with open(observer.status_file, "r") as f:
            events = [json.loads(line) for line in f]

        alert_events = [e for e in events if e["event"] == "slow_trial_alert"]
        assert len(alert_events) == 1

    def test_get_errors(self, observer):
        """get_errors returns all recorded errors."""
        observer.sweep_started(total_studies=2, algorithms=["dqn"])

        observer.study_started("study1", 1, 50)
        observer.study_failed("study1", "Error 1")

        observer.study_started("study2", 2, 50)
        observer.study_failed("study2", "Error 2")

        errors = observer.get_errors()
        assert len(errors) == 2
        assert errors[0]["study"] == "study1"
        assert errors[1]["study"] == "study2"


class TestStudyMetrics:
    """Tests for StudyMetrics dataclass."""

    def test_study_metrics_creation(self):
        """StudyMetrics can be created with all fields."""
        metrics = StudyMetrics(
            study_name="test",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            total_trials=50,
            completed_trials=45,
            pruned_trials=5,
            failed_trials=0,
            best_score=2000.0,
            best_params={"lr": 0.001},
            total_duration_seconds=3600.0,
            avg_trial_duration_seconds=80.0,
            status="completed",
        )

        assert metrics.study_name == "test"
        assert metrics.best_score == 2000.0

    def test_study_metrics_optional_error(self):
        """StudyMetrics error_message is optional."""
        metrics = StudyMetrics(
            study_name="test",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            total_trials=50,
            completed_trials=0,
            pruned_trials=0,
            failed_trials=1,
            best_score=0.0,
            best_params={},
            total_duration_seconds=0.0,
            avg_trial_duration_seconds=0.0,
            status="failed",
            error_message="Something went wrong",
        )

        assert metrics.error_message == "Something went wrong"
