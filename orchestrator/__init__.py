"""
Training Orchestrator Package.

The orchestrator manages multiple training runs across different
algorithm/representation/reward combinations, collects metrics,
and generates comparative reports.

Per Milestone 6 (design/milestones.md):
- Config-driven experiment launching
- Parallel run management
- Metrics collection and aggregation
- Report generation

Key modules:
- config: Experiment and orchestrator configuration dataclasses
- runner: ExperimentRunner for launching and managing training runs
- metrics: MetricsCollector for aggregating and storing metrics
- report: ReportGenerator for creating comparison reports
- cli: Command-line interface
"""

from orchestrator.config import ExperimentConfig, OrchestratorConfig
from orchestrator.runner import ExperimentRunner
from orchestrator.metrics import MetricsCollector
from orchestrator.report import ReportGenerator

__all__ = [
    "ExperimentConfig",
    "OrchestratorConfig",
    "ExperimentRunner",
    "MetricsCollector",
    "ReportGenerator",
]
