"""
Eligibility Traces for SARSA(lambda).

Eligibility traces track which state-action pairs were recently visited
and should receive credit for the current reward/TD-error.

Supports:
- Replacing traces: e(s,a) = 1 on visit (standard for control)
- Accumulating traces: e(s,a) += 1 on visit (less common)

Per DEC-0039: Vectorized tensor operations.
"""

from typing import Dict, Tuple, Optional
import torch
from torch import Tensor


class EligibilityTraces:
    """Eligibility traces for TD(lambda) methods.

    For neural network function approximation, we track traces for
    network parameters rather than state-action pairs directly.

    The traces decay by gamma*lambda after each step:
    e <- gamma * lambda * e

    On taking action a in state s:
    - Replacing: e = nabla_theta Q(s,a)
    - Accumulating: e += nabla_theta Q(s,a)

    Update: theta += alpha * delta * e
    where delta is the TD error.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        gamma: float,
        lambda_: float,
        replacing: bool = True
    ):
        """Initialize eligibility traces.

        Args:
            network: The Q-network whose parameters we're tracking
            gamma: Discount factor
            lambda_: Trace decay parameter (0=TD(0), 1=MC)
            replacing: If True, use replacing traces; else accumulating
        """
        self.network = network
        self.gamma = gamma
        self.lambda_ = lambda_
        self.replacing = replacing

        # Initialize traces for each parameter
        self.traces: Dict[str, Tensor] = {}
        for name, param in network.named_parameters():
            self.traces[name] = torch.zeros_like(param)

    def reset(self) -> None:
        """Reset all traces to zero (called at episode start)."""
        for name in self.traces:
            self.traces[name].zero_()

    def decay(self) -> None:
        """Decay all traces by gamma * lambda."""
        decay_factor = self.gamma * self.lambda_
        for name in self.traces:
            self.traces[name].mul_(decay_factor)

    def update(self, loss: Tensor) -> None:
        """Update traces with current gradient.

        Should be called after computing gradients for Q(s,a).

        For replacing traces: e = gradient
        For accumulating traces: e += gradient

        Args:
            loss: Loss tensor (should be -Q(s,a) for maximization)
        """
        # First decay existing traces
        self.decay()

        # Compute gradients
        self.network.zero_grad()
        loss.backward(retain_graph=True)

        # Update traces with gradients
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                if self.replacing:
                    # Replacing traces: set to gradient
                    self.traces[name] = param.grad.clone()
                else:
                    # Accumulating traces: add gradient
                    self.traces[name].add_(param.grad)

    def apply_update(self, td_error: Tensor, learning_rate: float) -> None:
        """Apply TD update using traces.

        Update: theta += alpha * delta * e

        Args:
            td_error: Scalar TD error (averaged over batch)
            learning_rate: Learning rate alpha
        """
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if name in self.traces:
                    # theta += alpha * delta * e
                    param.add_(self.traces[name], alpha=learning_rate * td_error.item())


class BatchedEligibilityTraces:
    """Batched eligibility traces for parallel environments.

    For batched training with N parallel games, we maintain separate
    traces for each game. When a game ends, its traces are reset.

    This is more memory intensive but allows correct trace handling
    with vectorized environments.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        n_games: int,
        gamma: float,
        lambda_: float,
        device: torch.device,
        replacing: bool = True
    ):
        """Initialize batched eligibility traces.

        Args:
            network: The Q-network
            n_games: Number of parallel games
            gamma: Discount factor
            lambda_: Trace decay parameter
            device: PyTorch device
            replacing: If True, use replacing traces
        """
        self.network = network
        self.n_games = n_games
        self.gamma = gamma
        self.lambda_ = lambda_
        self.device = device
        self.replacing = replacing

        # For batched traces, we use a simplified approach:
        # Store traces per parameter but as (n_games, *param_shape)
        # This allows per-game reset and decay
        self.traces: Dict[str, Tensor] = {}
        self._init_traces()

    def _init_traces(self) -> None:
        """Initialize trace tensors."""
        for name, param in self.network.named_parameters():
            # Traces have shape (n_games, *param_shape)
            trace_shape = (self.n_games,) + param.shape
            self.traces[name] = torch.zeros(trace_shape, device=self.device)

    def reset_games(self, done_mask: Tensor) -> None:
        """Reset traces for completed games.

        Args:
            done_mask: (N,) boolean mask of completed games
        """
        for name in self.traces:
            # Zero out traces for done games
            self.traces[name][done_mask] = 0

    def decay(self) -> None:
        """Decay all traces by gamma * lambda."""
        decay_factor = self.gamma * self.lambda_
        for name in self.traces:
            self.traces[name].mul_(decay_factor)

    def update_from_gradients(
        self,
        gradients: Dict[str, Tensor],
        game_indices: Optional[Tensor] = None
    ) -> None:
        """Update traces with computed gradients.

        Args:
            gradients: Dictionary mapping param names to gradients
            game_indices: Optional indices specifying which games to update.
                         If None, updates all games.
        """
        # First decay existing traces
        self.decay()

        # Update traces with gradients
        for name, grad in gradients.items():
            if name in self.traces:
                if game_indices is None:
                    # Update all games with same gradient
                    if self.replacing:
                        self.traces[name][:] = grad.unsqueeze(0)
                    else:
                        self.traces[name] += grad.unsqueeze(0)
                else:
                    # Update specific games
                    if self.replacing:
                        self.traces[name][game_indices] = grad
                    else:
                        self.traces[name][game_indices] += grad

    def get_aggregated_traces(self) -> Dict[str, Tensor]:
        """Get traces aggregated across all games.

        Returns:
            Dictionary mapping param names to mean traces across games
        """
        return {name: trace.mean(dim=0) for name, trace in self.traces.items()}

    def apply_update(
        self,
        td_errors: Tensor,
        learning_rate: float
    ) -> None:
        """Apply TD update using traces.

        For batched updates:
        theta += alpha * mean(delta_i * e_i)

        Args:
            td_errors: (N,) TD errors per game
            learning_rate: Learning rate alpha
        """
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if name in self.traces:
                    # Compute weighted sum of traces
                    # traces: (N, *param_shape), td_errors: (N,)
                    # We want: mean over N of (td_error[i] * trace[i])
                    weighted_traces = self.traces[name] * td_errors.view(-1, *([1] * len(param.shape)))
                    update = weighted_traces.mean(dim=0)
                    param.add_(update, alpha=learning_rate)
