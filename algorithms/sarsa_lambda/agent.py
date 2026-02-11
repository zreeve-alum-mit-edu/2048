"""
SARSA(lambda) Agent with Eligibility Traces.

Implements SARSA with eligibility traces for multi-step credit assignment.
Traces allow TD errors to propagate to previously visited states,
bridging the gap between TD(0) and Monte Carlo methods.

SARSA(lambda) update rule (backward view):
- For each step: e <- gamma * lambda * e + gradient(Q(s,a))
- Update: theta += alpha * delta * e
where delta = r + gamma * Q(s',a') - Q(s,a)

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling (traces reset on episode end)
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
Per DEC-0039: Vectorized tensor operations
"""

from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from algorithms.sarsa_lambda.model import SARSALambdaNetwork
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class SARSALambdaAgent:
    """SARSA(lambda) Agent for playing 2048.

    Uses eligibility traces to propagate TD errors to previously
    visited states. Lambda controls the trace decay:
    - lambda=0: TD(0), equivalent to standard SARSA
    - lambda=1: Monte Carlo (full episode returns)
    - 0<lambda<1: Intermediate, balancing bias and variance

    This implementation uses replacing traces, which are standard
    for control problems.
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: List[int] = [256, 256],
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        lambda_: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        replacing_traces: bool = True,
    ):
        """Initialize SARSA(lambda) agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Learning rate alpha
            gamma: Discount factor
            lambda_: Eligibility trace decay parameter
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon after decay
            epsilon_decay_steps: Steps for linear epsilon decay
            replacing_traces: If True, use replacing traces (recommended)
        """
        self.device = device
        self.gamma = gamma
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.replacing_traces = replacing_traces

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        input_size = self.representation.output_shape()[0]

        # Q-Network
        self.network = SARSALambdaNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Note: We don't use a standard optimizer because traces modify parameters directly
        # Instead, we'll use manual parameter updates with traces

        # Eligibility traces (one per parameter)
        self.traces: Dict[str, Tensor] = {}
        self._init_traces()

        # Training state
        self.step_count = 0
        self.epsilon = epsilon_start
        self.hidden_layers = hidden_layers

    def _init_traces(self) -> None:
        """Initialize eligibility traces to zero."""
        for name, param in self.network.named_parameters():
            self.traces[name] = torch.zeros_like(param)

    def reset_traces(self) -> None:
        """Reset all eligibility traces to zero."""
        for name in self.traces:
            self.traces[name].zero_()

    def _compute_epsilon(self) -> float:
        """Compute current epsilon based on linear decay schedule."""
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_end
        fraction = self.step_count / self.epsilon_decay_steps
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(
        self,
        state: Tensor,
        valid_mask: Tensor,
        training: bool = True
    ) -> Tensor:
        """Select actions for a batch of states.

        Args:
            state: (N, 16, 17) current board states
            valid_mask: (N, 4) valid action masks
            training: If True, use epsilon-greedy; if False, greedy

        Returns:
            (N,) selected actions
        """
        batch_size = state.size(0)

        with torch.no_grad():
            repr_state = self.representation(state)

        if training:
            self.epsilon = self._compute_epsilon()

            random_mask = torch.rand(batch_size, device=self.device) < self.epsilon

            with torch.no_grad():
                q_values = self.network.get_action_values(repr_state, valid_mask)
                greedy_actions = q_values.argmax(dim=1)

            # Random valid actions
            probs = valid_mask.float()
            row_sums = probs.sum(dim=1, keepdim=True)
            no_valid = (row_sums == 0).squeeze(1)
            if no_valid.any():
                probs[no_valid, 0] = 1.0
                row_sums = probs.sum(dim=1, keepdim=True)
            probs = probs / row_sums
            random_actions = torch.multinomial(probs, 1).squeeze(1)

            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
            with torch.no_grad():
                q_values = self.network.get_action_values(repr_state, valid_mask)
                actions = q_values.argmax(dim=1)

        return actions

    def train_step(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        next_action: Tensor,
        done: Tensor,
        valid_mask: Tensor,
        next_valid_mask: Tensor
    ) -> Dict[str, float]:
        """Perform one SARSA(lambda) update with eligibility traces.

        SARSA(lambda) backward view:
        1. Decay traces: e <- gamma * lambda * e
        2. Update traces: e += gradient(Q(s,a)) [or e = gradient for replacing]
        3. Compute TD error: delta = r + gamma * Q(s',a') - Q(s,a)
        4. Update parameters: theta += alpha * delta * e

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards received
            next_state: (N, 16, 17) next states
            next_action: (N,) next actions (for SARSA)
            done: (N,) done flags
            valid_mask: (N, 4) valid masks for current states
            next_valid_mask: (N, 4) valid masks for next states

        Returns:
            Dict with training metrics
        """
        # Normalize rewards
        if reward.std() > 0:
            reward_norm = (reward - reward.mean()) / (reward.std() + 1e-8)
        else:
            reward_norm = reward

        # Transform states
        repr_state = self.representation(state)
        with torch.no_grad():
            repr_next_state = self.representation(next_state)

        # Current Q-value for the action taken (need gradient)
        q_values = self.network(repr_state)
        current_q = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Next Q-value (no gradient needed)
        with torch.no_grad():
            next_q_values = self.network(repr_next_state)
            next_q = next_q_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            next_q = torch.where(done, torch.zeros_like(next_q), next_q)

            # TD error: delta = r + gamma * Q(s',a') - Q(s,a)
            td_error = reward_norm + self.gamma * next_q - current_q.detach()

        # Mean TD error for the update (scalar)
        mean_td_error = td_error.mean()

        # Compute gradient of Q(s,a) w.r.t. parameters
        # We use the mean Q-value as the loss to get gradients
        loss = current_q.mean()

        self.network.zero_grad()
        loss.backward()

        # Update eligibility traces
        # 1. Decay existing traces
        decay_factor = self.gamma * self.lambda_
        for name in self.traces:
            self.traces[name].mul_(decay_factor)

        # 2. Add/set current gradients to traces
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                if self.replacing_traces:
                    # Replacing traces: set to gradient
                    self.traces[name] = param.grad.clone()
                else:
                    # Accumulating traces: add gradient
                    self.traces[name].add_(param.grad)

        # 3. Update parameters using traces
        # theta += alpha * delta * e
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if name in self.traces:
                    param.add_(self.traces[name], alpha=self.learning_rate * mean_td_error.item())

        # Reset traces for finished episodes
        # For simplicity, we reset all traces if any game finished
        # A more sophisticated approach would track per-game traces
        if done.any():
            # Scale traces by fraction of continuing games
            continuing_fraction = 1.0 - done.float().mean().item()
            for name in self.traces:
                self.traces[name].mul_(continuing_fraction)

        self.step_count += 1

        return {
            "loss": (current_q.detach() - (reward_norm + self.gamma * next_q)).pow(2).mean().item(),
            "q_mean": current_q.mean().item(),
            "td_error": mean_td_error.item(),
            "trace_norm": sum(t.norm().item() for t in self.traces.values()),
            "epsilon": self.epsilon,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            "hidden_layers": self.hidden_layers,
            "lambda": self.lambda_,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
        # Reset traces when loading
        self._init_traces()
