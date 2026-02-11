"""
MCTS+Learned Algorithm.

Milestone 25: Monte Carlo Tree Search with learned value/policy networks.

Uses the real game environment for dynamics but neural networks for
policy priors and value estimation.

Per DEC-0005: Algorithm modules MUST be self-contained in algorithms/<name>/
Per DEC-0006: run.py MUST implement train() and evaluate() interfaces
"""

from algorithms.mcts_learned.run import train, evaluate, TrainingResult, EvalResult

__all__ = ['train', 'evaluate', 'TrainingResult', 'EvalResult']
