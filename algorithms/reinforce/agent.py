"""
REINFORCE Agent.

Implements the vanilla REINFORCE (Monte Carlo Policy Gradient) algorithm.
This is the simplest policy gradient method with no baseline/critic.

Key characteristics:
- Collects full trajectories before updating
- Uses Monte Carlo returns (sum of discounted rewards from each step)
- High variance but unbiased gradient estimates
- Good for sanity checking / baseline comparisons

Per DEC-0034: Mask-based action selection
Per DEC-0003: Episode boundary handling
Per DEC-0033: Uses merge_reward only
Per DEC-0037: Supports multiple representations
"""

from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.optim as optim
from torch import Tensor

from algorithms.reinforce.model import PolicyNetwork
from representations.base import Representation
from representations.onehot import OneHotRepresentation


class REINFORCEAgent:
    """REINFORCE Agent for playing 2048.

    Algorithm:
    1. Collect trajectories using current policy
    2. Compute Monte Carlo returns: G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
    3. Update policy: gradient = sum_t (G_t * grad log pi(a_t|s_t))

    This basic version has no baseline, so variance can be high.
    """

    def __init__(
        self,
        device: torch.device,
        representation: Optional[Representation] = None,
        hidden_layers: list = [256, 256],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
    ):
        """Initialize REINFORCE agent.

        Args:
            device: PyTorch device
            representation: Representation module for state transformation.
                           If None, uses OneHotRepresentation.
            hidden_layers: Hidden layer sizes for the network
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for returns
        """
        self.device = device
        self.gamma = gamma

        # Representation module (DEC-0037)
        if representation is None:
            self.representation = OneHotRepresentation({}).to(device)
        else:
            self.representation = representation.to(device)

        # Get input size from representation
        input_size = self.representation.output_shape()[0]

        # Policy network
        self.policy_net = PolicyNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        # Trajectory storage (per-game)
        # Each game stores its own trajectory until episode ends
        self._trajectories: Dict[int, List[Tuple[Tensor, int, float]]] = {}
        self._completed_trajectories: List[List[Tuple[Tensor, int, float]]] = []

        # Training state
        self.step_count = 0

    def select_action(
        self,
        state: Tensor,
        valid_mask: Tensor,
        training: bool = True
    ) -> Tensor:
        """Select actions for a batch of states.

        For REINFORCE, we always sample from the policy distribution
        during training (stochastic policy).

        Args:
            state: (N, 16, 17) current board states
            valid_mask: (N, 4) valid action masks
            training: If True, sample; if False, take argmax (greedy)

        Returns:
            (N,) selected actions
        """
        batch_size = state.size(0)

        # Transform state through representation
        with torch.no_grad():
            repr_state = self.representation(state)

        # Get action probabilities
        with torch.no_grad():
            probs = self.policy_net.get_action_probs(repr_state, valid_mask)

        if training:
            # Sample from the distribution (stochastic policy)
            # Handle edge case where all probs are 0 (shouldn't happen with valid_mask)
            probs_safe = probs.clone()
            row_sums = probs_safe.sum(dim=1, keepdim=True)
            no_valid = (row_sums == 0).squeeze(1)
            if no_valid.any():
                # Fallback: uniform over valid actions
                probs_safe[no_valid] = valid_mask[no_valid].float()
                probs_safe[no_valid] = probs_safe[no_valid] / probs_safe[no_valid].sum(dim=1, keepdim=True)

            actions = torch.multinomial(probs_safe, 1).squeeze(1)
        else:
            # Greedy action selection (for evaluation)
            actions = probs.argmax(dim=1)

        return actions

    def store_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        game_indices: Optional[Tensor] = None
    ) -> int:
        """Store transitions for trajectory collection.

        Unlike DQN, REINFORCE needs complete trajectories.
        We store (state, action, reward) tuples and process them
        when episodes end.

        Args:
            state: (N, 16, 17) current states
            action: (N,) actions taken
            reward: (N,) rewards (merge_reward per DEC-0033)
            done: (N,) episode termination flags
            game_indices: Optional explicit game indices (defaults to 0..N-1)

        Returns:
            Number of completed trajectories available for training
        """
        batch_size = state.size(0)

        if game_indices is None:
            game_indices = torch.arange(batch_size, device=self.device)

        # Store transitions for each game
        for i in range(batch_size):
            game_idx = game_indices[i].item()

            # Initialize trajectory storage for new games
            if game_idx not in self._trajectories:
                self._trajectories[game_idx] = []

            # Store (state, action, reward) - keep on device
            self._trajectories[game_idx].append((
                state[i].clone(),
                action[i].item(),
                reward[i].item()
            ))

            # If episode done, move trajectory to completed list
            if done[i]:
                if self._trajectories[game_idx]:  # Non-empty trajectory
                    self._completed_trajectories.append(self._trajectories[game_idx])
                self._trajectories[game_idx] = []

        return len(self._completed_trajectories)

    def _compute_returns(self, trajectory: List[Tuple[Tensor, int, float]]) -> List[float]:
        """Compute discounted returns for a trajectory.

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

        Args:
            trajectory: List of (state, action, reward) tuples

        Returns:
            List of returns for each timestep
        """
        returns = []
        G = 0.0

        # Compute returns backwards
        for _, _, reward in reversed(trajectory):
            G = reward + self.gamma * G
            returns.insert(0, G)

        return returns

    def train_step(self, min_trajectories: int = 1) -> Optional[Dict[str, float]]:
        """Perform one training step using collected trajectories.

        REINFORCE update:
        loss = -sum_t (G_t * log pi(a_t|s_t))

        Args:
            min_trajectories: Minimum number of completed trajectories to train

        Returns:
            Dict with training metrics or None if not enough trajectories
        """
        if len(self._completed_trajectories) < min_trajectories:
            return None

        # Process all completed trajectories
        all_states = []
        all_actions = []
        all_returns = []

        for trajectory in self._completed_trajectories:
            # Compute returns for this trajectory
            returns = self._compute_returns(trajectory)

            for (state, action, _), G in zip(trajectory, returns):
                all_states.append(state)
                all_actions.append(action)
                all_returns.append(G)

        # Clear completed trajectories
        self._completed_trajectories = []

        if not all_states:
            return None

        # Stack into tensors
        states = torch.stack(all_states)  # (T, 16, 17)
        actions = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        returns = torch.tensor(all_returns, dtype=torch.float32, device=self.device)

        # Normalize returns for stability (optional but helps)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Transform states through representation
        repr_states = self.representation(states)

        # Compute valid masks for all states
        # For stored states, we need to recompute valid masks
        from game.moves import compute_valid_mask
        valid_masks = compute_valid_mask(states, self.device)

        # Get log probabilities
        log_probs = self.policy_net.get_action_log_probs(repr_states, valid_masks)

        # Select log probs for taken actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Policy gradient loss: -sum(G * log_prob)
        loss = -(returns * action_log_probs).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "avg_return": returns.mean().item(),
            "num_transitions": len(all_states),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            # Save architecture info for loading
            "hidden_layers": [layer.out_features for layer in self.policy_net.network
                             if isinstance(layer, torch.nn.Linear)][:-1],
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
