"""
C51 (Categorical DQN) Algorithm.

Milestone 22: Distributional DQN that learns the distribution of returns
rather than just the expected value.

Per DEC-0005: Algorithm modules MUST be self-contained in algorithms/<name>/
Per DEC-0006: run.py MUST implement train() and evaluate() interfaces
"""

from algorithms.c51.run import train, evaluate, TrainingResult, EvalResult

__all__ = ['train', 'evaluate', 'TrainingResult', 'EvalResult']
