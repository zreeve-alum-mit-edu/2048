"""
Module entry point for sweep runner.

Usage:
    python -m sweep
    python -m sweep --status
    python -m sweep --dry-run
"""

from sweep.runner import main

if __name__ == "__main__":
    main()
