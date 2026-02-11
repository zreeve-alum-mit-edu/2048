"""
QR-DQN (Quantile Regression DQN) Algorithm.

Milestone 23: Distributional DQN using quantile regression.

Per DEC-0005: Algorithm modules MUST be self-contained in algorithms/<name>/
Per DEC-0006: run.py MUST implement train() and evaluate() interfaces
"""

from algorithms.qr_dqn.run import train, evaluate, TrainingResult, EvalResult

__all__ = ['train', 'evaluate', 'TrainingResult', 'EvalResult']
