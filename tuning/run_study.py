"""
CLI to Run a Single Optuna Study.

Usage:
    python -m tuning.run_study --study dqn_onehot_merge
    python -m tuning.run_study --study dqn_onehot_merge --trials 10 --n-jobs 2

Per DEC-0010: Optuna with SQLite storage.
Per DEC-0012: MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5).
Per DEC-0037: 50 trials, 4 parallel by default.
"""

import argparse
import os
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from tuning.study_config import STUDY_CONFIGS
from tuning.objective import create_objective


def main():
    """Run a single Optuna study."""
    parser = argparse.ArgumentParser(description="Run a single DQN tuning study")
    parser.add_argument(
        "--study",
        required=True,
        choices=list(STUDY_CONFIGS.keys()),
        help="Name of the study to run"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials (default: from config)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel trials (default: from config)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="SQLite storage path (default: from config)"
    )
    args = parser.parse_args()

    # Get study configuration
    config = STUDY_CONFIGS[args.study]

    # Override with command-line arguments
    n_trials = args.trials if args.trials is not None else config.n_trials
    n_jobs = args.n_jobs if args.n_jobs is not None else config.n_parallel_trials
    storage_path = args.storage if args.storage is not None else config.storage_path

    # Ensure storage directory exists
    if storage_path.startswith("sqlite:///"):
        db_path = storage_path.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Study: {config.study_name}")
    print(f"Representation: {config.representation_type}")
    print(f"Reward: {config.reward_type}")
    print(f"Trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Epochs per trial: {config.epochs_per_trial}")
    print(f"Steps per epoch: {config.steps_per_epoch}")
    print(f"Storage: {storage_path}")
    print(f"=" * 60)

    # Create pruner per DEC-0012
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=5
    )

    # Create sampler
    sampler = TPESampler(seed=42)

    # Create or load study
    study = optuna.create_study(
        study_name=config.study_name,
        storage=storage_path,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True
    )

    print(f"Loaded study with {len(study.trials)} existing trials")

    # Create objective function
    objective = create_objective(config)

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    # Print results
    print(f"\n" + "=" * 60)
    print(f"Study Complete: {config.study_name}")
    print(f"=" * 60)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (avg score): {study.best_value:.2f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params to file
    results_dir = Path("data/optuna/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{config.study_name}_best.txt"
    with open(results_file, "w") as f:
        f.write(f"Study: {config.study_name}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.2f}\n")
        f.write(f"\nBest hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
