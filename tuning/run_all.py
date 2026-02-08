"""
CLI to Run All 10 Optuna Studies.

Usage:
    python -m tuning.run_all
    python -m tuning.run_all --trials 10 --n-jobs 2
    python -m tuning.run_all --sequential

Per DEC-0009: Experimental matrix is (Algorithm, Representation, Reward) tuples.
Per DEC-0011: Each combo gets its own study.
Per DEC-0037: 10 studies total (5 representations x 2 rewards).
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tuning.study_config import STUDY_CONFIGS


def run_single_study(study_name: str, trials: int, n_jobs: int, storage: str) -> str:
    """Run a single study as a subprocess.

    Args:
        study_name: Name of the study
        trials: Number of trials
        n_jobs: Number of parallel trials
        storage: Storage path

    Returns:
        Result message
    """
    cmd = [
        sys.executable, "-m", "tuning.run_study",
        "--study", study_name,
        "--trials", str(trials),
        "--n-jobs", str(n_jobs),
        "--storage", storage
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=86400  # 24 hour timeout per study
        )
        if result.returncode == 0:
            return f"SUCCESS: {study_name}"
        else:
            return f"FAILED: {study_name}\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {study_name}"
    except Exception as e:
        return f"ERROR: {study_name}: {str(e)}"


def main():
    """Run all 10 Optuna studies."""
    parser = argparse.ArgumentParser(description="Run all DQN tuning studies")
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per study (default: 50)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel trials per study (default: 4)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///data/optuna/dqn_tuning.db",
        help="SQLite storage path"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run studies sequentially instead of in parallel"
    )
    parser.add_argument(
        "--max-parallel-studies",
        type=int,
        default=1,
        help="Maximum number of studies to run in parallel (default: 1)"
    )
    args = parser.parse_args()

    # Ensure storage directory exists
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    study_names = list(STUDY_CONFIGS.keys())

    print(f"=" * 60)
    print(f"Running All DQN Tuning Studies")
    print(f"=" * 60)
    print(f"Studies: {len(study_names)}")
    print(f"Trials per study: {args.trials}")
    print(f"Parallel trials per study: {args.n_jobs}")
    print(f"Storage: {args.storage}")
    print(f"Sequential: {args.sequential}")
    print(f"=" * 60)
    print()

    results = []

    if args.sequential or args.max_parallel_studies == 1:
        # Run studies one at a time
        for i, study_name in enumerate(study_names, 1):
            print(f"\n[{i}/{len(study_names)}] Running study: {study_name}")
            result = run_single_study(
                study_name, args.trials, args.n_jobs, args.storage
            )
            results.append(result)
            print(result)
    else:
        # Run multiple studies in parallel
        with ProcessPoolExecutor(max_workers=args.max_parallel_studies) as executor:
            future_to_study = {
                executor.submit(
                    run_single_study,
                    name, args.trials, args.n_jobs, args.storage
                ): name
                for name in study_names
            }

            for future in as_completed(future_to_study):
                study_name = future_to_study[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(result)
                except Exception as e:
                    results.append(f"ERROR: {study_name}: {str(e)}")
                    print(f"ERROR: {study_name}: {str(e)}")

    # Print summary
    print(f"\n" + "=" * 60)
    print(f"Summary")
    print(f"=" * 60)
    successes = sum(1 for r in results if r.startswith("SUCCESS"))
    failures = len(results) - successes
    print(f"Successful: {successes}/{len(study_names)}")
    print(f"Failed: {failures}/{len(study_names)}")

    if failures > 0:
        print(f"\nFailed studies:")
        for r in results:
            if not r.startswith("SUCCESS"):
                print(f"  {r}")


if __name__ == "__main__":
    main()
