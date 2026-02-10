"""
Experiment Runner.

Launches and manages training runs for algorithm experiments.

Per Milestone 6: Config-driven experiment launching, parallel run management.
Per DEC-0005: Algorithm modules in algorithms/<name>/
Per DEC-0006: Required train/evaluate interface
"""

import importlib
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from orchestrator.config import ExperimentConfig, OrchestratorConfig
from orchestrator.metrics import (
    EvaluationMetrics,
    ExperimentResult,
    MetricsCollector,
    TrainingMetrics,
)


def _create_env_factory(n_envs: int) -> Callable:
    """Create an environment factory callable.

    This is defined at module level so it can be pickled for multiprocessing.

    Args:
        n_envs: Number of parallel environments

    Returns:
        Factory callable that creates GameEnv instances
    """
    def factory():
        from game.env import GameEnv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return GameEnv(n_games=n_envs, device=device)
    return factory


def _run_single_experiment(
    exp_dict: dict,
    results_dir: str,
) -> dict:
    """Run a single experiment in a subprocess.

    This function is designed to be called via ProcessPoolExecutor.
    Takes dicts instead of dataclasses for pickle compatibility.

    Args:
        exp_dict: ExperimentConfig as dict
        results_dir: Base results directory

    Returns:
        ExperimentResult as dict
    """
    exp = ExperimentConfig(**exp_dict)

    try:
        # Import algorithm module dynamically
        module_path = f"algorithms.{exp.algorithm}.run"
        try:
            algo_module = importlib.import_module(module_path)
        except ImportError as e:
            return {
                "experiment_name": exp.name,
                "status": "failed",
                "error_message": f"Failed to import algorithm module '{module_path}': {e}",
            }

        # Verify required functions exist
        if not hasattr(algo_module, "train"):
            return {
                "experiment_name": exp.name,
                "status": "failed",
                "error_message": f"Algorithm module '{module_path}' missing required 'train' function",
            }
        if not hasattr(algo_module, "evaluate"):
            return {
                "experiment_name": exp.name,
                "status": "failed",
                "error_message": f"Algorithm module '{module_path}' missing required 'evaluate' function",
            }

        # Create environment factory
        env_factory = _create_env_factory(exp.n_envs)

        # Determine config path
        config_path = f"algorithms/{exp.algorithm}/config.yaml"
        if not Path(config_path).exists():
            return {
                "experiment_name": exp.name,
                "status": "failed",
                "error_message": f"Algorithm config not found: {config_path}",
            }

        # Run training
        start_time = time.time()
        training_result = algo_module.train(env_factory, config_path)
        training_time = time.time() - start_time

        # Get checkpoint for evaluation
        checkpoint_path = None
        if training_result.checkpoints:
            # Prefer final checkpoint
            for ckpt in reversed(training_result.checkpoints):
                if "final" in ckpt:
                    checkpoint_path = ckpt
                    break
            if checkpoint_path is None:
                checkpoint_path = training_result.checkpoints[-1]

        # Run evaluation
        eval_result = algo_module.evaluate(
            env_factory,
            checkpoint_path,
            exp.eval_games,
        )

        # Build metrics
        training_metrics = TrainingMetrics(
            experiment_name=exp.name,
            algorithm=exp.algorithm,
            representation=exp.representation,
            reward_type=exp.reward_type,
            total_steps=training_result.metrics.get("total_steps", 0),
            total_episodes=training_result.metrics.get("total_episodes", 0),
            final_avg_score=training_result.metrics.get("final_avg_score", 0.0),
            losses=training_result.metrics.get("losses", [])[-100:],  # Keep last 100
            q_means=training_result.metrics.get("q_means", [])[-100:],
            epsilons=training_result.metrics.get("epsilons", [])[-100:],
            eval_scores=training_result.metrics.get("eval_scores", []),
            training_time_seconds=training_time,
        )

        evaluation_metrics = EvaluationMetrics.from_scores(
            experiment_name=exp.name,
            scores=eval_result.scores,
        )

        return {
            "experiment_name": exp.name,
            "status": "success",
            "training_metrics": asdict(training_metrics),
            "evaluation_metrics": asdict(evaluation_metrics),
            "checkpoint_path": checkpoint_path,
        }

    except Exception as e:
        return {
            "experiment_name": exp.name,
            "status": "failed",
            "error_message": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


class ExperimentRunner:
    """Runs and manages training experiments.

    Supports sequential and parallel execution of experiments defined
    in an OrchestratorConfig.

    Attributes:
        config: Orchestrator configuration
        metrics_collector: Collector for experiment results
    """

    def __init__(self, config: OrchestratorConfig):
        """Initialize the experiment runner.

        Args:
            config: Orchestrator configuration with experiments
        """
        self.config = config
        self.metrics_collector = MetricsCollector(config.results_dir)

        # Ensure results directory exists
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    def _load_algorithm_module(self, algorithm_name: str):
        """Dynamically load an algorithm module.

        Args:
            algorithm_name: Name of algorithm (maps to algorithms/<name>/)

        Returns:
            Loaded module

        Raises:
            ImportError: If module cannot be loaded
        """
        module_path = f"algorithms.{algorithm_name}.run"
        return importlib.import_module(module_path)

    def _create_env_factory(self, exp: ExperimentConfig) -> Callable:
        """Create an environment factory for an experiment.

        Args:
            exp: Experiment configuration

        Returns:
            Callable that creates GameEnv instances
        """
        return _create_env_factory(exp.n_envs)

    def run_experiment(self, exp: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment.

        Args:
            exp: Experiment configuration

        Returns:
            ExperimentResult with training and evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting experiment: {exp.name}")
        print(f"  Algorithm: {exp.algorithm}")
        print(f"  Representation: {exp.representation}")
        print(f"  Reward type: {exp.reward_type}")
        print(f"{'='*60}\n")

        result_dict = _run_single_experiment(
            exp_dict={
                "name": exp.name,
                "algorithm": exp.algorithm,
                "representation": exp.representation,
                "reward_type": exp.reward_type,
                "training_steps": exp.training_steps,
                "eval_games": exp.eval_games,
                "n_envs": exp.n_envs,
                "hyperparameters": exp.hyperparameters,
                "checkpoint_dir": exp.checkpoint_dir,
            },
            results_dir=self.config.results_dir,
        )

        # Convert dict back to ExperimentResult
        result = self._dict_to_result(result_dict)

        # Store result
        self.metrics_collector.add_result(result)

        # Print summary
        if result.status == "success":
            print(f"\nExperiment {exp.name} completed successfully!")
            if result.evaluation_metrics:
                print(f"  Avg Score: {result.evaluation_metrics.avg_score:.1f}")
                print(f"  Max Score: {result.evaluation_metrics.max_score}")
        else:
            print(f"\nExperiment {exp.name} failed!")
            print(f"  Error: {result.error_message}")

        return result

    def _dict_to_result(self, result_dict: dict) -> ExperimentResult:
        """Convert result dict to ExperimentResult.

        Args:
            result_dict: Result as dictionary

        Returns:
            ExperimentResult instance
        """
        training_metrics = None
        evaluation_metrics = None

        if result_dict.get("training_metrics"):
            training_metrics = TrainingMetrics(**result_dict["training_metrics"])

        if result_dict.get("evaluation_metrics"):
            evaluation_metrics = EvaluationMetrics(**result_dict["evaluation_metrics"])

        return ExperimentResult(
            experiment_name=result_dict["experiment_name"],
            status=result_dict["status"],
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            error_message=result_dict.get("error_message"),
            checkpoint_path=result_dict.get("checkpoint_path"),
        )

    def run_all(self) -> List[ExperimentResult]:
        """Run all experiments in configuration.

        Respects parallel_runs setting for concurrent execution.

        Returns:
            List of ExperimentResults for all experiments
        """
        experiments = self.config.experiments
        results: List[ExperimentResult] = []

        print(f"\nStarting orchestrator run with {len(experiments)} experiments")
        print(f"Parallel runs: {self.config.parallel_runs}")
        print(f"Results directory: {self.config.results_dir}")

        if self.config.parallel_runs <= 1:
            # Sequential execution
            for exp in experiments:
                result = self.run_experiment(exp)
                results.append(result)
        else:
            # Parallel execution
            print(f"\nRunning experiments in parallel ({self.config.parallel_runs} workers)")

            # Convert experiments to dicts for pickling
            exp_dicts = []
            for exp in experiments:
                exp_dicts.append({
                    "name": exp.name,
                    "algorithm": exp.algorithm,
                    "representation": exp.representation,
                    "reward_type": exp.reward_type,
                    "training_steps": exp.training_steps,
                    "eval_games": exp.eval_games,
                    "n_envs": exp.n_envs,
                    "hyperparameters": exp.hyperparameters,
                    "checkpoint_dir": exp.checkpoint_dir,
                })

            with ProcessPoolExecutor(max_workers=self.config.parallel_runs) as executor:
                future_to_exp = {
                    executor.submit(
                        _run_single_experiment,
                        exp_dict,
                        self.config.results_dir,
                    ): exp_dict["name"]
                    for exp_dict in exp_dicts
                }

                for future in as_completed(future_to_exp):
                    exp_name = future_to_exp[future]
                    try:
                        result_dict = future.result()
                        result = self._dict_to_result(result_dict)
                    except Exception as e:
                        result = ExperimentResult(
                            experiment_name=exp_name,
                            status="failed",
                            error_message=f"Process error: {e}",
                        )

                    self.metrics_collector.add_result(result)
                    results.append(result)

                    if result.status == "success":
                        print(f"Completed: {exp_name}")
                    else:
                        print(f"Failed: {exp_name} - {result.error_message[:100]}...")

        # Print final summary
        successful = sum(1 for r in results if r.status == "success")
        failed = len(results) - successful

        print(f"\n{'='*60}")
        print(f"Orchestrator run complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"{'='*60}")

        return results

    def get_results(self) -> List[ExperimentResult]:
        """Get all collected results.

        Returns:
            List of ExperimentResults
        """
        return list(self.metrics_collector.results.values())
