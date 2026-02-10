"""
Report Generator.

Generates comparative reports from experiment results.

Per Milestone 6: Report generation with comparative metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.metrics import ExperimentResult, MetricsCollector


class ReportGenerator:
    """Generates comparative reports from experiment results.

    Supports markdown and JSON output formats.

    Attributes:
        results_dir: Directory containing experiment results
        metrics_collector: Collector with loaded results
    """

    def __init__(self, results_dir: str):
        """Initialize report generator.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.metrics_collector = MetricsCollector(results_dir)
        self.metrics_collector.load_existing_results()

    def _rank_experiments(
        self,
        results: List[ExperimentResult],
        sort_by: str = "avg_score",
    ) -> List[Dict[str, Any]]:
        """Rank experiments by performance metric.

        Args:
            results: List of successful experiment results
            sort_by: Metric to sort by (avg_score, max_score)

        Returns:
            List of ranked experiment data dicts
        """
        ranked = []

        for result in results:
            if result.status != "success" or result.evaluation_metrics is None:
                continue

            metrics = result.evaluation_metrics
            training = result.training_metrics

            entry = {
                "rank": 0,  # Will be set after sorting
                "name": result.experiment_name,
                "avg_score": metrics.avg_score,
                "max_score": metrics.max_score,
                "min_score": metrics.min_score,
                "std_score": metrics.std_score,
                "median_score": metrics.median_score,
                "num_games": metrics.num_games,
                "checkpoint_path": result.checkpoint_path,
            }

            if training:
                entry["algorithm"] = training.algorithm
                entry["representation"] = training.representation
                entry["reward_type"] = training.reward_type
                entry["training_time_seconds"] = training.training_time_seconds
                entry["total_steps"] = training.total_steps
                entry["total_episodes"] = training.total_episodes

            ranked.append(entry)

        # Sort by specified metric (descending)
        ranked.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

        # Assign ranks
        for i, entry in enumerate(ranked, 1):
            entry["rank"] = i

        return ranked

    def generate_markdown(self) -> str:
        """Generate a markdown report.

        Returns:
            Markdown-formatted report string
        """
        comparison = self.metrics_collector.get_comparison_data()
        results = list(self.metrics_collector.results.values())
        ranked = self._rank_experiments(results)

        lines = []

        # Header
        lines.append("# Training Orchestrator Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Experiments:** {comparison['total_experiments']}")
        lines.append(f"- **Successful:** {comparison['successful']}")
        lines.append(f"- **Failed:** {comparison['failed']}")
        lines.append("")

        # Rankings table
        if ranked:
            lines.append("## Experiment Rankings")
            lines.append("")
            lines.append("Ranked by average score (higher is better).")
            lines.append("")

            # Table header
            lines.append("| Rank | Experiment | Algorithm | Repr | Reward | Avg Score | Max Score | Std | Games |")
            lines.append("|------|------------|-----------|------|--------|-----------|-----------|-----|-------|")

            for entry in ranked:
                lines.append(
                    f"| {entry['rank']} "
                    f"| {entry['name']} "
                    f"| {entry.get('algorithm', 'N/A')} "
                    f"| {entry.get('representation', 'N/A')} "
                    f"| {entry.get('reward_type', 'N/A')} "
                    f"| {entry['avg_score']:.1f} "
                    f"| {entry['max_score']} "
                    f"| {entry['std_score']:.1f} "
                    f"| {entry['num_games']} |"
                )

            lines.append("")

            # Top performer highlight
            if ranked:
                top = ranked[0]
                lines.append("### Top Performer")
                lines.append("")
                lines.append(f"**{top['name']}**")
                lines.append("")
                lines.append(f"- Average Score: {top['avg_score']:.1f}")
                lines.append(f"- Maximum Score: {top['max_score']}")
                lines.append(f"- Standard Deviation: {top['std_score']:.1f}")
                if top.get('training_time_seconds'):
                    minutes = top['training_time_seconds'] / 60
                    lines.append(f"- Training Time: {minutes:.1f} minutes")
                if top.get('checkpoint_path'):
                    lines.append(f"- Checkpoint: `{top['checkpoint_path']}`")
                lines.append("")

        # Per-experiment details
        lines.append("## Experiment Details")
        lines.append("")

        for result in results:
            lines.append(f"### {result.experiment_name}")
            lines.append("")
            lines.append(f"**Status:** {result.status}")
            lines.append("")

            if result.status == "success":
                if result.training_metrics:
                    tm = result.training_metrics
                    lines.append("**Training:**")
                    lines.append(f"- Algorithm: {tm.algorithm}")
                    lines.append(f"- Representation: {tm.representation}")
                    lines.append(f"- Reward Type: {tm.reward_type}")
                    lines.append(f"- Total Steps: {tm.total_steps:,}")
                    lines.append(f"- Total Episodes: {tm.total_episodes:,}")
                    lines.append(f"- Training Time: {tm.training_time_seconds:.1f}s")
                    lines.append("")

                if result.evaluation_metrics:
                    em = result.evaluation_metrics
                    lines.append("**Evaluation:**")
                    lines.append(f"- Games Played: {em.num_games}")
                    lines.append(f"- Average Score: {em.avg_score:.1f}")
                    lines.append(f"- Maximum Score: {em.max_score}")
                    lines.append(f"- Minimum Score: {em.min_score}")
                    lines.append(f"- Median Score: {em.median_score:.1f}")
                    lines.append(f"- Std Deviation: {em.std_score:.1f}")
                    lines.append("")

                if result.checkpoint_path:
                    lines.append(f"**Checkpoint:** `{result.checkpoint_path}`")
                    lines.append("")

            else:
                lines.append(f"**Error:** {result.error_message}")
                lines.append("")

        # Failed experiments section
        failed = self.metrics_collector.get_failed_results()
        if failed:
            lines.append("## Failed Experiments")
            lines.append("")
            for result in failed:
                lines.append(f"- **{result.experiment_name}**: {result.error_message}")
            lines.append("")

        return "\n".join(lines)

    def generate_json(self) -> Dict[str, Any]:
        """Generate a JSON report.

        Returns:
            Report data as dictionary
        """
        comparison = self.metrics_collector.get_comparison_data()
        results = list(self.metrics_collector.results.values())
        ranked = self._rank_experiments(results)

        report = {
            "generated_at": datetime.now().isoformat(),
            "results_dir": str(self.results_dir),
            "summary": {
                "total_experiments": comparison["total_experiments"],
                "successful": comparison["successful"],
                "failed": comparison["failed"],
            },
            "rankings": ranked,
            "experiments": {},
        }

        for result in results:
            exp_data = {
                "status": result.status,
                "error_message": result.error_message,
                "checkpoint_path": result.checkpoint_path,
            }

            if result.training_metrics:
                tm = result.training_metrics
                exp_data["training"] = {
                    "algorithm": tm.algorithm,
                    "representation": tm.representation,
                    "reward_type": tm.reward_type,
                    "total_steps": tm.total_steps,
                    "total_episodes": tm.total_episodes,
                    "final_avg_score": tm.final_avg_score,
                    "training_time_seconds": tm.training_time_seconds,
                    "timestamp": tm.timestamp,
                }

            if result.evaluation_metrics:
                em = result.evaluation_metrics
                exp_data["evaluation"] = {
                    "num_games": em.num_games,
                    "avg_score": em.avg_score,
                    "max_score": em.max_score,
                    "min_score": em.min_score,
                    "median_score": em.median_score,
                    "std_score": em.std_score,
                    "scores": em.scores,
                    "timestamp": em.timestamp,
                }

            report["experiments"][result.experiment_name] = exp_data

        return report

    def save_reports(
        self,
        output_dir: Optional[str] = None,
        formats: List[str] = None,
    ) -> Dict[str, str]:
        """Save reports to files.

        Args:
            output_dir: Output directory (defaults to results_dir)
            formats: List of formats to generate (markdown, json, both)

        Returns:
            Dict mapping format to file path
        """
        if output_dir is None:
            output_dir = str(self.results_dir)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["markdown", "json"]

        saved = {}

        if "markdown" in formats or "both" in formats:
            md_content = self.generate_markdown()
            md_path = output_path / "report.md"
            with open(md_path, "w") as f:
                f.write(md_content)
            saved["markdown"] = str(md_path)
            print(f"Saved markdown report: {md_path}")

        if "json" in formats or "both" in formats:
            json_content = self.generate_json()
            json_path = output_path / "report.json"
            with open(json_path, "w") as f:
                json.dump(json_content, f, indent=2)
            saved["json"] = str(json_path)
            print(f"Saved JSON report: {json_path}")

        return saved
