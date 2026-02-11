"""
MuZero-style Algorithm.

Milestone 24: Model-based planning with learned dynamics.

Uses learned representation, dynamics, and prediction networks
for planning without requiring the real environment during search.

Per DEC-0005: Algorithm modules MUST be self-contained in algorithms/<name>/
Per DEC-0006: run.py MUST implement train() and evaluate() interfaces
"""

from algorithms.muzero_style.run import train, evaluate, TrainingResult, EvalResult

__all__ = ['train', 'evaluate', 'TrainingResult', 'EvalResult']
