"""
RL Algorithm Modules.

Per DEC-0005: Each algorithm is self-contained in algorithms/<name>/
with run.py as the required entry point.
"""

# Pre-load torch._dynamo to avoid 10+ second delay on first optimizer creation.
# PyTorch 2.x lazily initializes dynamo when an optimizer is first created,
# which causes unexpected delays during agent initialization. By importing
# it here at module load time, the cost is paid once during import.
import torch._dynamo  # noqa: F401
