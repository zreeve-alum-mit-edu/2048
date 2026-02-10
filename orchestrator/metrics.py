"""
Metrics Collection and Aggregation.

Collects and stores metrics from training experiments.

Per Milestone 6: Metrics collection and aggregation.
Per existing patterns: Uses JSONL format consistent with context/observability.jsonl
"""

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class TrainingMetrics:
    """Metrics collected during training.

    Attributes:
        experiment_name: Name of the experiment
        algorithm: Algorithm used
        representation: Representation type
        reward_type: Reward signal used
        total_steps: Total training steps completed
        total_episodes: Total episodes completed
        final_avg_score: Average score at end of training
        losses: List of training losses (sampled)
        q_means: List of mean Q-values (sampled)
        epsilons: List of epsilon values (sampled)
        eval_scores: Evaluation scores during training
        training_time_seconds: Total training time
        timestamp: When training completed
    """
    experiment_name: str
    algorithm: str
    representation: str
    reward_type: str
    total_steps: int = 0
    total_episodes: int = 0
    final_avg_score: float = 0.0
    losses: List[float] = field(default_factory=list)
    q_means: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    eval_scores: List[Dict[str, Any]] = field(default_factory=list)
    training_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationMetrics:
    """Metrics from final evaluation.

    Attributes:
        experiment_name: Name of the experiment
        num_games: Number of games played
        scores: List of individual game scores
        avg_score: Mean score
        max_score: Maximum score achieved
        min_score: Minimum score
        std_score: Standard deviation of scores
        median_score: Median score
        timestamp: When evaluation completed
    """
    experiment_name: str
    num_games: int
    scores: List[int]
    avg_score: float
    max_score: int
    min_score: int
    std_score: float
    median_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_scores(cls, experiment_name: str, scores: List[int]) -> "EvaluationMetrics":
        """Create EvaluationMetrics from a list of scores.

        Args:
            experiment_name: Name of the experiment
            scores: List of game scores

        Returns:
            EvaluationMetrics with computed statistics
        """
        if not scores:
            return cls(
                experiment_name=experiment_name,
                num_games=0,
                scores=[],
                avg_score=0.0,
                max_score=0,
                min_score=0,
                std_score=0.0,
                median_score=0.0,
            )

        return cls(
            experiment_name=experiment_name,
            num_games=len(scores),
            scores=scores,
            avg_score=statistics.mean(scores),
            max_score=max(scores),
            min_score=min(scores),
            std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            median_score=statistics.median(scores),
        )


@dataclass
class ExperimentResult:
    """Complete result of an experiment.

    Combines training and evaluation metrics with status information.

    Attributes:
        experiment_name: Name of the experiment
        status: "success", "failed", or "skipped"
        training_metrics: Metrics from training (if successful)
        evaluation_metrics: Metrics from evaluation (if successful)
        error_message: Error message (if failed)
        checkpoint_path: Path to final checkpoint (if successful)
    """
    experiment_name: str
    status: str
    training_metrics: Optional[TrainingMetrics] = None
    evaluation_metrics: Optional[EvaluationMetrics] = None
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None


class MetricsCollector:
    """Collects and aggregates metrics from multiple experiments.

    Provides methods to add results, compute aggregates, and save to files.
    Uses JSONL format for storage.

    Attributes:
        results_dir: Base directory for storing results
        results: Dict mapping experiment names to results
    """

    def __init__(self, results_dir: str):
        """Initialize metrics collector.

        Args:
            results_dir: Base directory for storing results
        """
        self.results_dir = Path(results_dir)
        self.results: Dict[str, ExperimentResult] = {}

    def add_result(self, result: ExperimentResult) -> None:
        """Add an experiment result.

        Args:
            result: ExperimentResult to add
        """
        self.results[result.experiment_name] = result
        self._save_result(result)

    def _save_result(self, result: ExperimentResult) -> None:
        """Save a single result to its experiment directory.

        Args:
            result: Result to save
        """
        exp_dir = self.results_dir / result.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save training metrics
        if result.training_metrics:
            metrics_file = exp_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(asdict(result.training_metrics), f, indent=2)

        # Save evaluation metrics
        if result.evaluation_metrics:
            eval_file = exp_dir / "evaluation_metrics.json"
            with open(eval_file, 'w') as f:
                json.dump(asdict(result.evaluation_metrics), f, indent=2)

        # Append to global results JSONL
        results_file = self.results_dir / "all_results.jsonl"
        with open(results_file, 'a') as f:
            record = {
                "experiment_name": result.experiment_name,
                "status": result.status,
                "error_message": result.error_message,
                "checkpoint_path": result.checkpoint_path,
                "timestamp": datetime.now().isoformat(),
            }
            if result.evaluation_metrics:
                record["avg_score"] = result.evaluation_metrics.avg_score
                record["max_score"] = result.evaluation_metrics.max_score
            f.write(json.dumps(record) + "\n")

    def get_result(self, experiment_name: str) -> Optional[ExperimentResult]:
        """Get result for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            ExperimentResult if found, None otherwise
        """
        return self.results.get(experiment_name)

    def get_successful_results(self) -> List[ExperimentResult]:
        """Get all successful experiment results.

        Returns:
            List of successful ExperimentResults
        """
        return [r for r in self.results.values() if r.status == "success"]

    def get_failed_results(self) -> List[ExperimentResult]:
        """Get all failed experiment results.

        Returns:
            List of failed ExperimentResults
        """
        return [r for r in self.results.values() if r.status == "failed"]

    def get_comparison_data(self) -> Dict[str, Any]:
        """Get data for comparison report.

        Returns:
            Dict with aggregated comparison data
        """
        successful = self.get_successful_results()

        if not successful:
            return {
                "total_experiments": len(self.results),
                "successful": 0,
                "failed": len(self.results),
                "experiments": [],
            }

        # Extract comparison data
        experiments = []
        for result in successful:
            if result.evaluation_metrics:
                experiments.append({
                    "name": result.experiment_name,
                    "avg_score": result.evaluation_metrics.avg_score,
                    "max_score": result.evaluation_metrics.max_score,
                    "min_score": result.evaluation_metrics.min_score,
                    "std_score": result.evaluation_metrics.std_score,
                    "num_games": result.evaluation_metrics.num_games,
                    "checkpoint_path": result.checkpoint_path,
                })

        # Sort by avg_score descending
        experiments.sort(key=lambda x: x["avg_score"], reverse=True)

        return {
            "total_experiments": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "experiments": experiments,
        }

    def load_existing_results(self) -> None:
        """Load existing results from results directory.

        Scans for training_metrics.json and evaluation_metrics.json files.
        """
        if not self.results_dir.exists():
            return

        for exp_dir in self.results_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            experiment_name = exp_dir.name
            training_metrics = None
            evaluation_metrics = None

            # Load training metrics
            training_file = exp_dir / "training_metrics.json"
            if training_file.exists():
                with open(training_file, 'r') as f:
                    data = json.load(f)
                    training_metrics = TrainingMetrics(**data)

            # Load evaluation metrics
            eval_file = exp_dir / "evaluation_metrics.json"
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                    evaluation_metrics = EvaluationMetrics(**data)

            if training_metrics or evaluation_metrics:
                # Find checkpoint if exists
                checkpoint_path = None
                checkpoints_dir = exp_dir / "checkpoints"
                if checkpoints_dir.exists():
                    final_ckpt = checkpoints_dir / "dqn_final.pt"
                    if final_ckpt.exists():
                        checkpoint_path = str(final_ckpt)

                result = ExperimentResult(
                    experiment_name=experiment_name,
                    status="success",
                    training_metrics=training_metrics,
                    evaluation_metrics=evaluation_metrics,
                    checkpoint_path=checkpoint_path,
                )
                self.results[experiment_name] = result

    def summary(self) -> str:
        """Get a summary string of all results.

        Returns:
            Human-readable summary
        """
        lines = ["Metrics Summary", "=" * 40]

        successful = self.get_successful_results()
        failed = self.get_failed_results()

        lines.append(f"Total experiments: {len(self.results)}")
        lines.append(f"Successful: {len(successful)}")
        lines.append(f"Failed: {len(failed)}")

        if successful:
            lines.append("")
            lines.append("Top performers by avg score:")
            sorted_results = sorted(
                successful,
                key=lambda r: r.evaluation_metrics.avg_score if r.evaluation_metrics else 0,
                reverse=True
            )
            for i, result in enumerate(sorted_results[:5], 1):
                if result.evaluation_metrics:
                    lines.append(
                        f"  {i}. {result.experiment_name}: "
                        f"avg={result.evaluation_metrics.avg_score:.1f}, "
                        f"max={result.evaluation_metrics.max_score}"
                    )

        return "\n".join(lines)
