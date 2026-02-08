"""
Optuna Objective Function.

Implements the training objective for DQN hyperparameter tuning.

Per DEC-0012: MedianPruner with epoch-level reporting.
Per DEC-0037: 300 epochs, 2500 steps/epoch, 50 eval games.
"""

from typing import Callable, Optional
import optuna
from optuna import Trial

import torch

from game.env import GameEnv
from game.moves import compute_valid_mask
from algorithms.dqn.agent import DQNAgent
from tuning.study_config import StudyConfig
from tuning.search_spaces import suggest_hyperparams
from tuning.utils import create_representation


def create_objective(config: StudyConfig) -> Callable[[Trial], float]:
    """Create Optuna objective function for a study.

    Args:
        config: Study configuration

    Returns:
        Objective function that takes a Trial and returns float (avg score)
    """

    def objective(trial: Trial) -> float:
        """Train DQN and return final evaluation score.

        Args:
            trial: Optuna trial object

        Returns:
            Average score over final evaluation games
        """
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Suggest hyperparameters
        params = suggest_hyperparams(trial, config.representation_type)

        # Create representation
        representation = create_representation(config.representation_type, params)
        representation = representation.to(device)

        # Create agent with suggested hyperparameters
        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params["epsilon_start"],
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
        )

        # Create environment
        env = GameEnv(n_games=32, device=device)

        # Training loop with epoch-level reporting
        state = env.reset()
        total_reward = torch.zeros(env.n_games, device=device)

        for epoch in range(config.epochs_per_trial):
            # Train for steps_per_epoch steps
            epoch_scores = []

            for step in range(config.steps_per_epoch):
                # Get valid actions
                valid_mask = compute_valid_mask(state, device)

                # Select actions
                actions = agent.select_action(state, valid_mask, training=True)

                # Environment step
                result = env.step(actions)

                # Get reward based on reward type
                if config.reward_type == "merge":
                    reward = result.merge_reward.float()
                else:  # spawn
                    reward = result.spawn_reward.float()

                # Store transition
                agent.store_transition(
                    state=state,
                    action=actions,
                    reward=reward,
                    next_state=result.next_state,
                    done=result.done,
                    valid_mask=result.valid_mask,
                )

                # Track scores
                total_reward += reward

                # Handle episode termination
                if result.done.any():
                    for i in range(env.n_games):
                        if result.done[i]:
                            epoch_scores.append(total_reward[i].item())
                            total_reward[i] = 0.0

                # Train step
                agent.train_step()

                # Update state
                state = torch.where(
                    result.done.unsqueeze(-1).unsqueeze(-1),
                    result.reset_states,
                    result.next_state
                )
                env._state = state.clone()

            # Evaluate at end of epoch
            eval_score = _quick_eval(agent, device, config.eval_games_per_epoch)

            # Report to Optuna for pruning
            trial.report(eval_score, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # Final evaluation with more games
        final_score = _quick_eval(agent, device, num_games=100)
        return final_score

    return objective


def _quick_eval(
    agent: DQNAgent,
    device: torch.device,
    num_games: int,
    batch_size: int = 32
) -> float:
    """Quick evaluation of agent.

    Args:
        agent: DQN agent
        device: PyTorch device
        num_games: Number of games to evaluate
        batch_size: Batch size for parallel evaluation

    Returns:
        Average score over all games
    """
    env = GameEnv(n_games=batch_size, device=device)
    scores = []
    games_completed = 0

    while games_completed < num_games:
        state = env.reset()
        episode_scores = torch.zeros(batch_size, device=device)
        done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        while not done_mask.all():
            valid_mask = compute_valid_mask(state, device)

            # Greedy action selection
            actions = agent.select_action(state, valid_mask, training=False)

            result = env.step(actions)

            # Accumulate scores (using merge_reward for consistent evaluation)
            episode_scores += torch.where(
                done_mask,
                torch.zeros_like(result.merge_reward.float()),
                result.merge_reward.float()
            )

            # Track newly completed games
            newly_done = result.done & ~done_mask
            for i in range(batch_size):
                if newly_done[i] and games_completed < num_games:
                    scores.append(episode_scores[i].item())
                    games_completed += 1

            done_mask = done_mask | result.done

            # Update state
            state = torch.where(
                result.done.unsqueeze(-1).unsqueeze(-1),
                result.reset_states,
                result.next_state
            )
            env._state = state.clone()

    return sum(scores[:num_games]) / num_games if scores else 0.0
