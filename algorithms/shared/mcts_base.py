"""
Base MCTS Implementation.

Provides a parameterized MCTS algorithm that can be used by:
- MCTS+learned: Uses real environment for dynamics
- MuZero-style: Uses learned dynamics model

The key difference is how dynamics are handled (real env vs learned model),
which is abstracted through the DynamicsProvider protocol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from algorithms.shared.tree import TreeNode


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.25
    exploration_fraction: float = 0.25
    temperature: float = 1.0
    temperature_drop_step: int = 50000


class DynamicsProvider(ABC):
    """Abstract interface for dynamics (real or learned).

    This allows MCTS to work with either:
    - Real environment (MCTS+learned)
    - Learned dynamics model (MuZero)
    """

    @abstractmethod
    def get_initial_state(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        """Get initial state and valid actions from observation.

        Args:
            observation: (16, 17) one-hot game state

        Returns:
            Tuple of:
            - state/hidden_state tensor
            - valid_mask (4,) boolean
        """
        pass

    @abstractmethod
    def step(
        self,
        state: Tensor,
        action: int
    ) -> Tuple[Tensor, float, bool, Tensor]:
        """Take a step in the dynamics.

        Args:
            state: Current state/hidden state
            action: Action to take

        Returns:
            Tuple of:
            - next_state: Next state/hidden state
            - reward: Reward received
            - done: Whether episode ended
            - valid_mask: (4,) valid actions in next state
        """
        pass

    @abstractmethod
    def get_policy_value(
        self,
        state: Tensor,
        valid_mask: Tensor
    ) -> Tuple[Tensor, float]:
        """Get policy and value estimate for state.

        Args:
            state: Current state/hidden state
            valid_mask: (4,) valid actions mask

        Returns:
            Tuple of:
            - policy: (4,) action probabilities
            - value: State value estimate
        """
        pass


class MCTSBase:
    """Base MCTS algorithm implementation.

    Performs Monte Carlo Tree Search using a dynamics provider
    for state transitions and a policy-value network for guidance.
    """

    def __init__(
        self,
        dynamics_provider: DynamicsProvider,
        config: MCTSConfig,
        device: torch.device,
    ):
        """Initialize MCTS.

        Args:
            dynamics_provider: Provider for dynamics (real or learned)
            config: MCTS configuration
            device: PyTorch device
        """
        self.dynamics = dynamics_provider
        self.config = config
        self.device = device

    def search(
        self,
        observation: Tensor,
        add_exploration_noise: bool = True,
    ) -> TreeNode:
        """Run MCTS from given observation.

        Args:
            observation: (16, 17) game state
            add_exploration_noise: Whether to add Dirichlet noise at root

        Returns:
            Root node with search results
        """
        # Get initial state and valid actions
        state, valid_mask = self.dynamics.get_initial_state(observation)

        # Get initial policy and value
        policy, value = self.dynamics.get_policy_value(state, valid_mask)

        # Create root node
        root = TreeNode()

        # Add Dirichlet noise to root prior for exploration
        if add_exploration_noise and self.config.exploration_fraction > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((4,), self.config.dirichlet_alpha, device=self.device)
            ).sample()

            # Only apply noise to valid actions
            noisy_policy = policy.clone()
            for a in range(4):
                if valid_mask[a]:
                    noisy_policy[a] = (
                        (1 - self.config.exploration_fraction) * policy[a] +
                        self.config.exploration_fraction * noise[a]
                    )

            # Renormalize over valid actions
            valid_sum = noisy_policy[valid_mask].sum()
            if valid_sum > 0:
                noisy_policy[valid_mask] = noisy_policy[valid_mask] / valid_sum
            policy = noisy_policy

        # Expand root
        root.expand(policy, valid_mask, state=state, hidden_state=state)
        root.visit_count = 1
        root.value_sum = value

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root)

        return root

    def _simulate(self, root: TreeNode) -> None:
        """Run single MCTS simulation from root.

        Args:
            root: Root node to start simulation from
        """
        node = root
        search_path = [node]

        # Selection: traverse tree to leaf
        while node.is_expanded():
            action, node = node.select_child(self.config.c_puct)
            search_path.append(node)

        # Get parent state for dynamics step
        parent = node.parent
        if parent is None:
            # This shouldn't happen if root is expanded
            return

        parent_state = parent.hidden_state if parent.hidden_state is not None else parent.state

        # Expansion: use dynamics to get next state
        action = node.action_from_parent
        next_state, reward, done, valid_mask = self.dynamics.step(
            parent_state, action
        )

        if done:
            # Terminal node - value is 0
            value = 0.0
            # Create a minimal expansion for terminal state
            node.state = next_state
            node.hidden_state = next_state
            node.reward = reward
            node.valid_actions = valid_mask
        else:
            # Get policy and value for new state
            policy, value = self.dynamics.get_policy_value(next_state, valid_mask)

            # Expand the node
            node.expand(
                policy, valid_mask,
                state=next_state,
                hidden_state=next_state,
                reward=reward
            )

        # Backpropagation
        self._backpropagate(search_path, value, reward)

    def _backpropagate(
        self,
        search_path: list,
        value: float,
        leaf_reward: float
    ) -> None:
        """Backpropagate value through search path.

        Args:
            search_path: List of nodes from root to leaf
            value: Value estimate at leaf
            leaf_reward: Reward received at leaf
        """
        # For simplicity, we use a single value backprop
        # More sophisticated versions could use n-step returns
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            # Optionally incorporate reward into value
            # value = leaf_reward + gamma * value  # if doing n-step

    def select_action(
        self,
        root: TreeNode,
        temperature: Optional[float] = None
    ) -> int:
        """Select action from search results.

        Args:
            root: Root node with completed search
            temperature: Temperature for selection (None = use config)

        Returns:
            Selected action
        """
        if temperature is None:
            temperature = self.config.temperature

        return root.select_action_by_visit_count(temperature)

    def get_policy_target(self, root: TreeNode) -> Tensor:
        """Get policy target (visit count distribution) for training.

        Args:
            root: Root node with completed search

        Returns:
            (4,) policy target based on visit counts
        """
        return root.get_action_distribution()
