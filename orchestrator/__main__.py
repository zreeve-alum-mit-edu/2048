"""
Training Orchestrator CLI.

Entry point for running experiments and generating reports.

Usage:
    python -m orchestrator run <config.yaml>
    python -m orchestrator report <results_dir>

Per Milestone 6: CLI entry point for orchestrator.
"""

import argparse
import sys
from pathlib import Path

from orchestrator.config import load_config, create_quick_config, save_config
from orchestrator.runner import ExperimentRunner
from orchestrator.report import ReportGenerator


def cmd_run(args):
    """Run experiments from configuration file.

    Args:
        args: Parsed command line arguments
    """
    config_path = args.config

    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    print(f"Loaded configuration from: {config_path}")
    print(f"  Experiments: {len(config.experiments)}")
    print(f"  Parallel runs: {config.parallel_runs}")
    print(f"  Results dir: {config.results_dir}")

    # Override results_dir if specified
    if args.results_dir:
        config.results_dir = args.results_dir

    # Override parallel_runs if specified
    if args.parallel is not None:
        config.parallel_runs = args.parallel

    # Run experiments
    runner = ExperimentRunner(config)
    results = runner.run_all()

    # Generate reports if requested
    if args.report:
        generator = ReportGenerator(config.results_dir)
        generator.save_reports(formats=[config.report_format])

    # Print summary
    successful = sum(1 for r in results if r.status == "success")
    failed = len(results) - successful

    print(f"\nFinal Summary:")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")

    if failed > 0:
        sys.exit(1)


def cmd_report(args):
    """Generate reports from results directory.

    Args:
        args: Parsed command line arguments
    """
    results_dir = args.results_dir

    if not Path(results_dir).exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    generator = ReportGenerator(results_dir)

    # Determine output directory
    output_dir = args.output or results_dir

    # Determine formats
    formats = []
    if args.format == "both":
        formats = ["markdown", "json"]
    else:
        formats = [args.format]

    saved = generator.save_reports(output_dir=output_dir, formats=formats)

    print(f"\nReports generated:")
    for fmt, path in saved.items():
        print(f"  {fmt}: {path}")


def cmd_quick(args):
    """Generate a quick comparison config.

    Args:
        args: Parsed command line arguments
    """
    algorithms = args.algorithms.split(",")
    representations = args.representations.split(",")
    rewards = args.rewards.split(",")

    config = create_quick_config(
        algorithms=algorithms,
        representations=representations,
        reward_types=rewards,
        training_steps=args.steps,
        eval_games=args.eval_games,
        results_dir=args.results_dir,
    )

    output_path = args.output
    save_config(config, output_path)

    print(f"Generated config with {len(config.experiments)} experiments:")
    for exp in config.experiments:
        print(f"  - {exp.name}")
    print(f"\nSaved to: {output_path}")


def cmd_list(args):
    """List experiments in a config file.

    Args:
        args: Parsed command line arguments
    """
    config_path = args.config

    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print(f"Configuration: {config_path}")
    print(f"  Parallel runs: {config.parallel_runs}")
    print(f"  Results dir: {config.results_dir}")
    print(f"  Report format: {config.report_format}")
    print(f"\nExperiments ({len(config.experiments)}):")

    for exp in config.experiments:
        print(f"  - {exp.name}")
        print(f"      Algorithm: {exp.algorithm}")
        print(f"      Representation: {exp.representation}")
        print(f"      Reward: {exp.reward_type}")
        print(f"      Steps: {exp.training_steps}")
        print(f"      Eval games: {exp.eval_games}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Training Orchestrator for 2048 RL Experiments",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run experiments from config file",
    )
    run_parser.add_argument(
        "config",
        help="Path to YAML configuration file",
    )
    run_parser.add_argument(
        "--results-dir",
        help="Override results directory",
    )
    run_parser.add_argument(
        "--parallel",
        type=int,
        help="Override parallel runs setting",
    )
    run_parser.add_argument(
        "--report",
        action="store_true",
        help="Generate reports after completion",
    )
    run_parser.set_defaults(func=cmd_run)

    # report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate reports from results directory",
    )
    report_parser.add_argument(
        "results_dir",
        help="Path to results directory",
    )
    report_parser.add_argument(
        "--output", "-o",
        help="Output directory (defaults to results_dir)",
    )
    report_parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json", "both"],
        default="both",
        help="Report format (default: both)",
    )
    report_parser.set_defaults(func=cmd_report)

    # quick command
    quick_parser = subparsers.add_parser(
        "quick",
        help="Generate a quick comparison config",
    )
    quick_parser.add_argument(
        "--algorithms", "-a",
        required=True,
        help="Comma-separated list of algorithms",
    )
    quick_parser.add_argument(
        "--representations", "-r",
        required=True,
        help="Comma-separated list of representations",
    )
    quick_parser.add_argument(
        "--rewards", "-w",
        default="merge",
        help="Comma-separated list of reward types (default: merge)",
    )
    quick_parser.add_argument(
        "--steps", "-s",
        type=int,
        default=100000,
        help="Training steps per experiment (default: 100000)",
    )
    quick_parser.add_argument(
        "--eval-games", "-e",
        type=int,
        default=100,
        help="Evaluation games per experiment (default: 100)",
    )
    quick_parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory (default: results)",
    )
    quick_parser.add_argument(
        "--output", "-o",
        default="quick_config.yaml",
        help="Output config file (default: quick_config.yaml)",
    )
    quick_parser.set_defaults(func=cmd_quick)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List experiments in a config file",
    )
    list_parser.add_argument(
        "config",
        help="Path to YAML configuration file",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
