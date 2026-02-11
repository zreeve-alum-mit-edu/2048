"""
Observability Module for Sweep Execution.

Provides comprehensive logging, status tracking, and monitoring capabilities
for long-running experimental sweeps.

Key features:
- Real-time status file updates (JSONL format, can be tailed)
- Progress tracking (studies queued/running/completed)
- Performance monitoring (time per trial/study, ETAs)
- Error capture and reporting without crashing sweep
- Incremental results logging
"""

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque


@dataclass
class TrialMetrics:
    """Metrics for a single trial."""
    trial_number: int
    score: float
    duration_seconds: float
    pruned: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyMetrics:
    """Metrics for a completed study."""
    study_name: str
    algorithm: str
    representation: str
    reward_type: str
    total_trials: int
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    best_score: float
    best_params: Dict[str, Any]
    total_duration_seconds: float
    avg_trial_duration_seconds: float
    status: str  # "completed", "failed", "running"
    error_message: Optional[str] = None


class SweepObserver:
    """Observability layer for sweep execution.

    Provides real-time monitoring, logging, and status tracking for
    long-running experimental sweeps.

    Attributes:
        status_file: Path to JSONL status file for real-time monitoring
        results_file: Path to JSONL results file for completed studies
        total_studies: Total number of studies in sweep
    """

    def __init__(
        self,
        status_file: str = "context/sweep_status.jsonl",
        results_file: str = "results/sweep_results.jsonl",
        summary_file: str = "results/sweep_summary.json",
        slow_trial_threshold_seconds: float = 600.0,  # 10 minutes
        slow_study_threshold_minutes: float = 120.0,  # 2 hours
    ):
        """Initialize sweep observer.

        Args:
            status_file: Path to status JSONL file
            results_file: Path to results JSONL file
            summary_file: Path to summary JSON file
            slow_trial_threshold_seconds: Alert threshold for slow trials
            slow_study_threshold_minutes: Alert threshold for slow studies
        """
        self.status_file = Path(status_file)
        self.results_file = Path(results_file)
        self.summary_file = Path(summary_file)
        self.slow_trial_threshold = slow_trial_threshold_seconds
        self.slow_study_threshold = slow_study_threshold_minutes * 60

        # Create directories
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.total_studies = 0
        self.completed_studies = 0
        self.failed_studies = 0
        self.current_study: Optional[str] = None
        self.current_study_start: Optional[float] = None
        self.current_trial: int = 0

        # Performance tracking
        self.trial_durations: deque = deque(maxlen=100)  # Rolling window
        self.study_durations: List[float] = []
        self.sweep_start_time: Optional[float] = None

        # Error tracking
        self.errors: List[Dict[str, Any]] = []

        # Thread safety
        self._lock = threading.Lock()

    def _log_event(self, event: Dict[str, Any]) -> None:
        """Write event to status file.

        Args:
            event: Event dictionary to log
        """
        event["ts"] = datetime.now().isoformat()
        with self._lock:
            with open(self.status_file, "a") as f:
                f.write(json.dumps(event) + "\n")

    def _log_result(self, result: Dict[str, Any]) -> None:
        """Write result to results file.

        Args:
            result: Result dictionary to log
        """
        result["ts"] = datetime.now().isoformat()
        with self._lock:
            with open(self.results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    def sweep_started(self, total_studies: int, algorithms: List[str]) -> None:
        """Log sweep start.

        Args:
            total_studies: Total number of studies
            algorithms: List of algorithms being swept
        """
        self.total_studies = total_studies
        self.sweep_start_time = time.time()
        self.completed_studies = 0
        self.failed_studies = 0

        self._log_event({
            "event": "sweep_started",
            "total_studies": total_studies,
            "algorithms": algorithms,
        })

        print(f"\n{'='*70}")
        print(f"SWEEP STARTED")
        print(f"{'='*70}")
        print(f"Total studies: {total_studies}")
        print(f"Algorithms: {len(algorithms)}")
        print(f"Status file: {self.status_file}")
        print(f"Results file: {self.results_file}")
        print(f"{'='*70}\n")

    def study_started(self, study_name: str, index: int, n_trials: int) -> None:
        """Log study start.

        Args:
            study_name: Name of study
            index: Study index (1-based)
            n_trials: Number of trials planned
        """
        self.current_study = study_name
        self.current_study_start = time.time()
        self.current_trial = 0

        self._log_event({
            "event": "study_started",
            "study": study_name,
            "index": index,
            "total": self.total_studies,
            "n_trials": n_trials,
        })

        progress_pct = (index - 1) / self.total_studies * 100
        eta = self._calculate_eta()

        print(f"\n[{index}/{self.total_studies}] Starting study: {study_name}")
        print(f"  Progress: {progress_pct:.1f}%")
        if eta:
            print(f"  ETA: {eta}")

    def trial_started(self, trial_number: int) -> None:
        """Log trial start.

        Args:
            trial_number: Trial number within study
        """
        self.current_trial = trial_number
        self._log_event({
            "event": "trial_started",
            "study": self.current_study,
            "trial": trial_number,
        })

    def trial_completed(
        self,
        trial_number: int,
        score: float,
        duration_seconds: float,
        params: Dict[str, Any],
        pruned: bool = False,
    ) -> None:
        """Log trial completion.

        Args:
            trial_number: Trial number
            score: Trial score
            duration_seconds: Trial duration
            params: Trial hyperparameters
            pruned: Whether trial was pruned
        """
        self.trial_durations.append(duration_seconds)

        event = {
            "event": "trial_completed",
            "study": self.current_study,
            "trial": trial_number,
            "score": score,
            "duration_seconds": round(duration_seconds, 2),
            "pruned": pruned,
        }
        self._log_event(event)

        # Check for slow trial
        if duration_seconds > self.slow_trial_threshold:
            self._log_event({
                "event": "slow_trial_alert",
                "study": self.current_study,
                "trial": trial_number,
                "duration_seconds": round(duration_seconds, 2),
                "threshold_seconds": self.slow_trial_threshold,
            })
            print(f"  [ALERT] Slow trial {trial_number}: {duration_seconds:.1f}s")

        # Progress output
        avg_duration = sum(self.trial_durations) / len(self.trial_durations)
        status = "PRUNED" if pruned else f"score={score:.1f}"
        print(f"  Trial {trial_number}: {status} ({duration_seconds:.1f}s, avg={avg_duration:.1f}s)")

    def study_completed(
        self,
        study_name: str,
        metrics: StudyMetrics,
    ) -> None:
        """Log study completion.

        Args:
            study_name: Study name
            metrics: Study metrics
        """
        self.completed_studies += 1
        if self.current_study_start:
            duration = time.time() - self.current_study_start
            self.study_durations.append(duration)
            metrics.total_duration_seconds = duration

        # Log event
        self._log_event({
            "event": "study_completed",
            "study": study_name,
            "completed": self.completed_studies,
            "total": self.total_studies,
            "best_score": metrics.best_score,
            "trials_completed": metrics.completed_trials,
            "trials_pruned": metrics.pruned_trials,
            "duration_seconds": round(metrics.total_duration_seconds, 2),
        })

        # Log result
        self._log_result({
            "study": study_name,
            "algorithm": metrics.algorithm,
            "representation": metrics.representation,
            "reward_type": metrics.reward_type,
            "status": "completed",
            "best_score": metrics.best_score,
            "best_params": metrics.best_params,
            "total_trials": metrics.total_trials,
            "completed_trials": metrics.completed_trials,
            "pruned_trials": metrics.pruned_trials,
            "duration_seconds": round(metrics.total_duration_seconds, 2),
        })

        # Update summary
        self._update_summary()

        # Check for slow study
        if metrics.total_duration_seconds > self.slow_study_threshold:
            self._log_event({
                "event": "slow_study_alert",
                "study": study_name,
                "duration_seconds": round(metrics.total_duration_seconds, 2),
                "threshold_seconds": self.slow_study_threshold,
            })

        progress_pct = self.completed_studies / self.total_studies * 100
        duration_mins = metrics.total_duration_seconds / 60

        print(f"\n  Study completed: {study_name}")
        print(f"  Best score: {metrics.best_score:.1f}")
        print(f"  Trials: {metrics.completed_trials} completed, {metrics.pruned_trials} pruned")
        print(f"  Duration: {duration_mins:.1f} minutes")
        print(f"  Overall progress: {self.completed_studies}/{self.total_studies} ({progress_pct:.1f}%)")

    def study_failed(self, study_name: str, error: str) -> None:
        """Log study failure.

        Args:
            study_name: Study name
            error: Error message
        """
        self.failed_studies += 1
        self.errors.append({
            "study": study_name,
            "error": error,
            "ts": datetime.now().isoformat(),
        })

        self._log_event({
            "event": "study_failed",
            "study": study_name,
            "error": error[:500],  # Truncate long errors
            "failed_count": self.failed_studies,
        })

        self._log_result({
            "study": study_name,
            "status": "failed",
            "error": error[:1000],
        })

        self._update_summary()

        print(f"\n  [ERROR] Study failed: {study_name}")
        print(f"  Error: {error[:200]}...")
        print(f"  Failed studies: {self.failed_studies}")

    def sweep_completed(self) -> None:
        """Log sweep completion."""
        total_duration = 0.0
        if self.sweep_start_time:
            total_duration = time.time() - self.sweep_start_time

        self._log_event({
            "event": "sweep_completed",
            "total_studies": self.total_studies,
            "completed": self.completed_studies,
            "failed": self.failed_studies,
            "duration_seconds": round(total_duration, 2),
        })

        self._update_summary()

        hours = total_duration / 3600

        print(f"\n{'='*70}")
        print(f"SWEEP COMPLETED")
        print(f"{'='*70}")
        print(f"Total studies: {self.total_studies}")
        print(f"Completed: {self.completed_studies}")
        print(f"Failed: {self.failed_studies}")
        print(f"Total duration: {hours:.1f} hours")
        print(f"Results: {self.results_file}")
        print(f"Summary: {self.summary_file}")
        print(f"{'='*70}\n")

    def _calculate_eta(self) -> Optional[str]:
        """Calculate estimated time of completion.

        Returns:
            ETA string or None if not enough data
        """
        if not self.study_durations:
            return None

        avg_study_duration = sum(self.study_durations) / len(self.study_durations)
        remaining = self.total_studies - self.completed_studies - self.failed_studies
        remaining_seconds = remaining * avg_study_duration

        eta = datetime.now() + timedelta(seconds=remaining_seconds)
        hours = remaining_seconds / 3600

        return f"{eta.strftime('%Y-%m-%d %H:%M')} ({hours:.1f} hours remaining)"

    def _update_summary(self) -> None:
        """Update summary JSON file."""
        total_duration = 0.0
        if self.sweep_start_time:
            total_duration = time.time() - self.sweep_start_time

        summary = {
            "last_updated": datetime.now().isoformat(),
            "status": "running" if self.completed_studies + self.failed_studies < self.total_studies else "completed",
            "total_studies": self.total_studies,
            "completed_studies": self.completed_studies,
            "failed_studies": self.failed_studies,
            "remaining_studies": self.total_studies - self.completed_studies - self.failed_studies,
            "progress_percent": round((self.completed_studies + self.failed_studies) / max(1, self.total_studies) * 100, 1),
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_hours": round(total_duration / 3600, 2),
            "avg_study_duration_seconds": round(sum(self.study_durations) / len(self.study_durations), 2) if self.study_durations else 0,
            "current_study": self.current_study,
            "current_trial": self.current_trial,
            "eta": self._calculate_eta(),
            "errors": self.errors[-10:],  # Last 10 errors
        }

        with self._lock:
            with open(self.summary_file, "w") as f:
                json.dump(summary, f, indent=2)

    def get_status(self) -> Dict[str, Any]:
        """Get current sweep status.

        Returns:
            Status dictionary
        """
        return {
            "total_studies": self.total_studies,
            "completed_studies": self.completed_studies,
            "failed_studies": self.failed_studies,
            "remaining_studies": self.total_studies - self.completed_studies - self.failed_studies,
            "current_study": self.current_study,
            "current_trial": self.current_trial,
            "progress_percent": round((self.completed_studies + self.failed_studies) / max(1, self.total_studies) * 100, 1),
            "eta": self._calculate_eta(),
        }

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all recorded errors.

        Returns:
            List of error dictionaries
        """
        return self.errors.copy()
