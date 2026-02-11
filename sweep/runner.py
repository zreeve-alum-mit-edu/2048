"""
Sweep Runner - Main Orchestrator for Milestone 26 Full Sweep.

Executes all 200 studies with comprehensive observability, error handling,
and resume capability.

Usage:
    python -m sweep.runner
    python -m sweep.runner --trials 10 --algorithms dqn double_dqn
    python -m sweep.runner --resume
    python -m sweep.runner --status

Per DEC-0021: Full experimental sweep covering all algorithms x representations x rewards.
Per DEC-0010: Optuna with SQLite storage (enables resume).
Per DEC-0012: MedianPruner configuration.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Set

import optuna
from optuna.samplers import TPESampler

from sweep.observability import SweepObserver, StudyMetrics
from sweep.study_factory import StudyFactory, SweepStudyConfig, ALGORITHMS
from sweep.objective_factory import ObjectiveFactory


class SweepRunner:
    """Main sweep orchestrator.

    Runs all studies with observability, error handling, and resume support.

    Attributes:
        factory: Study configuration factory
        observer: Observability layer
        obj_factory: Objective function factory
    """

    def __init__(
        self,
        n_trials: int = 50,
        epochs_per_trial: int = 300,
        steps_per_epoch: int = 2500,
        eval_games_per_epoch: int = 50,
        storage_path: str = "sqlite:///data/optuna/sweep.db",
        algorithms: Optional[List[str]] = None,
        representations: Optional[List[str]] = None,
        reward_types: Optional[List[str]] = None,
        status_file: str = "context/sweep_status.jsonl",
        results_file: str = "results/sweep_results.jsonl",
    ):
        """Initialize sweep runner.

        Args:
            n_trials: Trials per study
            epochs_per_trial: Epochs per trial
            steps_per_epoch: Steps per epoch
            eval_games_per_epoch: Eval games per epoch
            storage_path: Optuna storage path
            algorithms: List of algorithms (default: all)
            representations: List of representations (default: all)
            reward_types: List of reward types (default: both)
            status_file: Path to status JSONL file
            results_file: Path to results JSONL file
        """
        self.storage_path = storage_path

        # Create factory with specified parameters
        self.factory = StudyFactory(
            n_trials=n_trials,
            epochs_per_trial=epochs_per_trial,
            steps_per_epoch=steps_per_epoch,
            eval_games_per_epoch=eval_games_per_epoch,
            n_parallel_trials=1,  # Sequential for GPU memory
            storage_path=storage_path,
            algorithms=algorithms,
            representations=representations,
            reward_types=reward_types,
        )

        # Create observer
        self.observer = SweepObserver(
            status_file=status_file,
            results_file=results_file,
        )

        # Create objective factory with observer
        self.obj_factory = ObjectiveFactory(observer=self.observer)

        # Ensure storage directory exists
        if storage_path.startswith("sqlite:///"):
            db_path = storage_path.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def get_completed_studies(self) -> Set[str]:
        """Get set of already completed study names.

        Reads from results file to support resume.

        Returns:
            Set of completed study names
        """
        completed = set()
        results_path = Path(self.observer.results_file)

        if results_path.exists():
            with open(results_path, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record.get("status") == "completed":
                            completed.add(record["study"])
                    except json.JSONDecodeError:
                        continue

        return completed

    def run_study(self, config: SweepStudyConfig, index: int) -> bool:
        """Run a single study.

        Args:
            config: Study configuration
            index: Study index (1-based)

        Returns:
            True if successful, False if failed
        """
        self.observer.study_started(
            config.study_name,
            index,
            config.n_trials,
        )

        try:
            # Create pruner per DEC-0012
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5,
            )

            # Create sampler
            sampler = TPESampler(seed=42 + index)

            # Create or load study
            study = optuna.create_study(
                study_name=config.study_name,
                storage=config.storage_path,
                direction="maximize",
                pruner=pruner,
                sampler=sampler,
                load_if_exists=True,
            )

            existing_trials = len(study.trials)
            if existing_trials > 0:
                print(f"  Resuming study with {existing_trials} existing trials")

            # Create objective
            objective = self.obj_factory.create_objective(config)

            # Calculate remaining trials
            remaining_trials = max(0, config.n_trials - existing_trials)

            if remaining_trials == 0:
                print(f"  Study already completed with {existing_trials} trials")
            else:
                # Run optimization
                study.optimize(
                    objective,
                    n_trials=remaining_trials,
                    n_jobs=config.n_parallel_trials,
                    show_progress_bar=False,  # Using our own progress tracking
                )

            # Gather metrics
            completed_trials = len([t for t in study.trials
                                    if t.state == optuna.trial.TrialState.COMPLETE])
            pruned_trials = len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.PRUNED])
            failed_trials = len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.FAIL])

            # Calculate average trial duration
            trial_durations = []
            for trial in study.trials:
                if trial.datetime_complete and trial.datetime_start:
                    duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
                    trial_durations.append(duration)

            avg_trial_duration = sum(trial_durations) / len(trial_durations) if trial_durations else 0

            metrics = StudyMetrics(
                study_name=config.study_name,
                algorithm=config.algorithm,
                representation=config.representation,
                reward_type=config.reward_type,
                total_trials=len(study.trials),
                completed_trials=completed_trials,
                pruned_trials=pruned_trials,
                failed_trials=failed_trials,
                best_score=study.best_value if study.best_trial else 0.0,
                best_params=study.best_params if study.best_trial else {},
                total_duration_seconds=0,  # Will be set by observer
                avg_trial_duration_seconds=avg_trial_duration,
                status="completed",
            )

            self.observer.study_completed(config.study_name, metrics)
            return True

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self.observer.study_failed(config.study_name, error_msg)
            return False

    def run(self, resume: bool = True) -> None:
        """Run the full sweep.

        Args:
            resume: If True, skip already completed studies
        """
        configs = self.factory.generate_all_configs()

        # Get completed studies for resume
        completed = set()
        if resume:
            completed = self.get_completed_studies()
            if completed:
                print(f"Resuming sweep with {len(completed)} completed studies")

        # Filter out completed
        remaining_configs = [c for c in configs if c.study_name not in completed]

        if not remaining_configs:
            print("All studies already completed!")
            return

        # Start sweep
        self.observer.sweep_started(
            total_studies=len(configs),
            algorithms=self.factory.algorithms,
        )

        # Update observer with correct counts
        self.observer.completed_studies = len(completed)

        # Run studies
        for i, config in enumerate(remaining_configs, 1):
            global_index = configs.index(config) + 1
            success = self.run_study(config, global_index)

            # Brief pause between studies for cleanup
            time.sleep(1)

        self.observer.sweep_completed()

    def status(self) -> None:
        """Print current sweep status."""
        status = self.observer.get_status()
        summary_file = self.observer.summary_file

        print(f"\n{'='*50}")
        print("SWEEP STATUS")
        print(f"{'='*50}")

        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)

            print(f"Status: {summary.get('status', 'unknown')}")
            print(f"Progress: {summary.get('completed_studies', 0)}/{summary.get('total_studies', 0)} "
                  f"({summary.get('progress_percent', 0):.1f}%)")
            print(f"Failed: {summary.get('failed_studies', 0)}")
            print(f"Duration: {summary.get('total_duration_hours', 0):.1f} hours")
            print(f"Current: {summary.get('current_study', 'N/A')}")
            print(f"ETA: {summary.get('eta', 'N/A')}")

            errors = summary.get("errors", [])
            if errors:
                print(f"\nRecent Errors ({len(errors)}):")
                for err in errors[-3:]:
                    print(f"  - {err.get('study', 'unknown')}: {err.get('error', '')[:50]}...")
        else:
            print("No sweep in progress or completed")

        print(f"{'='*50}\n")


def main():
    """Main entry point for sweep runner."""
    parser = argparse.ArgumentParser(
        description="Run Milestone 26 Full Experimental Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full sweep (200 studies)
    python -m sweep.runner

    # Run with fewer trials for testing
    python -m sweep.runner --trials 5 --epochs 10

    # Run only specific algorithms
    python -m sweep.runner --algorithms dqn double_dqn a2c

    # Run only specific representations
    python -m sweep.runner --representations onehot embedding

    # Check status
    python -m sweep.runner --status

    # Resume interrupted sweep
    python -m sweep.runner --resume
        """
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per study (default: 50)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Epochs per trial (default: 300)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2500,
        help="Steps per epoch (default: 2500)"
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Evaluation games per epoch (default: 50)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///data/optuna/sweep.db",
        help="Optuna storage path"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        help="Algorithms to include (default: all)"
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=["onehot", "embedding", "cnn_2x2", "cnn_4x1", "cnn_multi"],
        help="Representations to include (default: all)"
    )
    parser.add_argument(
        "--rewards",
        nargs="+",
        choices=["merge", "spawn"],
        help="Reward types to include (default: both)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous run (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore previous results"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current sweep status and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )

    args = parser.parse_args()

    # Create runner
    runner = SweepRunner(
        n_trials=args.trials,
        epochs_per_trial=args.epochs,
        steps_per_epoch=args.steps,
        eval_games_per_epoch=args.eval_games,
        storage_path=args.storage,
        algorithms=args.algorithms,
        representations=args.representations,
        reward_types=args.rewards,
    )

    # Handle status command
    if args.status:
        runner.status()
        return

    # Handle dry run
    if args.dry_run:
        configs = runner.factory.generate_all_configs()
        summary = runner.factory.summary()

        print(f"\n{'='*50}")
        print("DRY RUN - SWEEP CONFIGURATION")
        print(f"{'='*50}")
        print(f"Algorithms: {len(summary['algorithms'])}")
        for algo in summary["algorithms"]:
            print(f"  - {algo}")
        print(f"Representations: {len(summary['representations'])}")
        for repr_type in summary["representations"]:
            print(f"  - {repr_type}")
        print(f"Reward types: {summary['reward_types']}")
        print(f"Total studies: {summary['total_studies']}")
        print(f"Trials per study: {summary['n_trials_per_study']}")
        print(f"Epochs per trial: {summary['epochs_per_trial']}")
        print(f"Steps per epoch: {summary['steps_per_epoch']}")
        print(f"Total trials: {summary['total_trials']}")
        print(f"{'='*50}\n")

        if args.resume:
            completed = runner.get_completed_studies()
            remaining = [c for c in configs if c.study_name not in completed]
            print(f"Completed studies: {len(completed)}")
            print(f"Remaining studies: {len(remaining)}")
        return

    # Run sweep
    resume = args.resume and not args.no_resume
    runner.run(resume=resume)


if __name__ == "__main__":
    main()
