"""
Root pytest configuration for 2048 RL project tests.

This module provides shared configuration and markers for all tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu_timing: marks tests that verify GPU timing thresholds"
    )
    config.addinivalue_line(
        "markers", "gpu_required: marks tests that require CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow-running"
    )


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-gpu-timing",
        action="store_true",
        default=False,
        help="Run GPU timing tests (requires calibrated thresholds)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU timing tests unless explicitly requested."""
    if not config.getoption("--run-gpu-timing"):
        skip_timing = pytest.mark.skip(
            reason="GPU timing tests skipped (use --run-gpu-timing to enable)"
        )
        for item in items:
            if "gpu_timing" in item.keywords:
                item.add_marker(skip_timing)
