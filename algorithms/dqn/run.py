"""
DQN Algorithm Entry Point.

Per DEC-0005: Algorithm modules MUST be self-contained in algorithms/<name>/
Per DEC-0006: run.py MUST implement train() and evaluate() with specified signatures

This module implements the required interface for DQN training and evaluation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional

import torch
import yaml

from algorithms.dqn.agent import DQNAgent
from game.env import GameEnv


@dataclass
class TrainingResult:
    """Result of training run.

    Per DEC-0006: Standardized result type.

    Attributes:
        checkpoints: List of saved checkpoint paths
        metrics: Algorithm-specific metrics (free-form)
    """
    checkpoints: List[str]
    metrics: Dict[str, Any]


@dataclass
class EvalResult:
    """Result of evaluation run.

    Per DEC-0006: Standardized result type.

    Attributes:
        scores: Final score per game
        avg_score: Average across all games
        max_score: Best score achieved
    """
    scores: List[int]
    avg_score: float
    max_score: int


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(
    env_factory: Callable[[], GameEnv],
    config_path: str
) -> TrainingResult:
    """Train DQN agent.

    Per DEC-0006: Required interface for algorithm training.

    Args:
        env_factory: Callable that returns a new GameEnv instance
        config_path: Path to algorithm-specific config file

    Returns:
        TrainingResult with checkpoint paths and training metrics
    """
    # Load configuration
    config = load_config(config_path)

    # Create environment first to get its device
    env = env_factory()
    device = env.device
    print(f"Using device: {device}")

    # Create agent
    agent = DQNAgent(
        device=device,
        hidden_layers=config["network"]["hidden_layers"],
        learning_rate=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        epsilon_start=config["epsilon"]["start"],
        epsilon_end=config["epsilon"]["end"],
        epsilon_decay_steps=config["epsilon"]["decay_steps"],
        target_update_frequency=config["target_network"]["update_frequency"],
        buffer_capacity=config["replay_buffer"]["capacity"],
        buffer_min_size=config["replay_buffer"]["min_size"],
        batch_size=config["training"]["batch_size"],
    )

    # Training metrics
    total_steps = config["training"]["total_steps"]
    log_frequency = config["logging"]["log_frequency"]
    eval_frequency = config["logging"]["eval_frequency"]
    eval_games = config["logging"]["eval_games"]
    checkpoint_frequency = config["checkpoint"]["save_frequency"]
    checkpoint_dir = Path(config["checkpoint"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = []
    metrics = {
        "losses": [],
        "q_means": [],
        "epsilons": [],
        "eval_scores": [],
        "episode_scores": [],
    }

    # Initialize environment
    state = env.reset()

    # Track episode scores per game
    episode_scores = torch.zeros(env.n_games, device=device)
    completed_episodes = 0
    total_episode_score = 0.0

    print(f"Starting training for {total_steps} steps...")

    step = 0
    while step < total_steps:
        # Get valid actions mask from environment
        # Note: We need to compute valid mask before action selection
        from game.moves import compute_valid_mask
        valid_mask = compute_valid_mask(state, device)

        # Select actions
        actions = agent.select_action(state, valid_mask, training=True)

        # Environment step
        result = env.step(actions)

        # Store transitions (DEC-0033: use merge_reward)
        agent.store_transition(
            state=state,
            action=actions,
            reward=result.merge_reward.float(),
            next_state=result.next_state,
            done=result.done,
            valid_mask=result.valid_mask,
        )

        # Track episode scores
        episode_scores += result.merge_reward.float()

        # Handle episode termination (vectorized per DEC-0039)
        if result.done.any():
            done_scores = episode_scores[result.done]
            num_done = done_scores.numel()
            completed_episodes += num_done
            total_episode_score += done_scores.sum().item()
            metrics["episode_scores"].extend(done_scores.tolist())
            episode_scores = torch.where(result.done, torch.zeros_like(episode_scores), episode_scores)

        # Train
        train_metrics = agent.train_step()

        if train_metrics is not None:
            metrics["losses"].append(train_metrics["loss"])
            metrics["q_means"].append(train_metrics["q_mean"])
            metrics["epsilons"].append(train_metrics["epsilon"])

        # Update state (handle episode boundaries per DEC-0003)
        # For done games, use reset_states; for others, use next_state
        state = torch.where(
            result.done.unsqueeze(-1).unsqueeze(-1),
            result.reset_states,
            result.next_state
        )

        # CRITICAL: Also update env's internal state to match
        # The env doesn't auto-reset done games, so we must sync manually
        env._state = state.clone()

        step += 1

        # Logging
        if step % log_frequency == 0:
            avg_loss = sum(metrics["losses"][-log_frequency:]) / log_frequency if metrics["losses"] else 0
            avg_q = sum(metrics["q_means"][-log_frequency:]) / log_frequency if metrics["q_means"] else 0
            eps = agent.epsilon
            avg_ep_score = total_episode_score / completed_episodes if completed_episodes > 0 else 0

            print(f"Step {step}/{total_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Q: {avg_q:.2f} | "
                  f"Epsilon: {eps:.3f} | "
                  f"Episodes: {completed_episodes} | "
                  f"Avg Score: {avg_ep_score:.1f}")

        # Evaluation
        if step % eval_frequency == 0:
            eval_result = evaluate(env_factory, None, eval_games, agent=agent)
            metrics["eval_scores"].append({
                "step": step,
                "avg_score": eval_result.avg_score,
                "max_score": eval_result.max_score,
            })
            print(f"  Eval: Avg={eval_result.avg_score:.1f}, Max={eval_result.max_score}")

        # Checkpointing
        if step % checkpoint_frequency == 0:
            checkpoint_path = str(checkpoint_dir / f"dqn_step_{step}.pt")
            agent.save_checkpoint(checkpoint_path)
            checkpoints.append(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = str(checkpoint_dir / "dqn_final.pt")
    agent.save_checkpoint(final_checkpoint)
    checkpoints.append(final_checkpoint)

    # Compute final metrics
    final_avg_score = total_episode_score / completed_episodes if completed_episodes > 0 else 0
    metrics["final_avg_score"] = final_avg_score
    metrics["total_episodes"] = completed_episodes
    metrics["total_steps"] = step

    print(f"\nTraining complete!")
    print(f"Total episodes: {completed_episodes}")
    print(f"Final avg score: {final_avg_score:.1f}")

    return TrainingResult(checkpoints=checkpoints, metrics=metrics)


def evaluate(
    env_factory: Callable[[], GameEnv],
    checkpoint_path: Optional[str],
    num_games: int,
    agent: Optional[DQNAgent] = None
) -> EvalResult:
    """Evaluate DQN agent.

    Per DEC-0006: Required interface for algorithm evaluation.

    Args:
        env_factory: Callable that returns a new GameEnv instance
        checkpoint_path: Path to saved model (None if agent provided)
        num_games: How many games to run
        agent: Optional pre-loaded agent (for mid-training evaluation)

    Returns:
        EvalResult with scores
    """
    # Create environment first to get its device
    env = env_factory()
    device = env.device
    batch_size = env.n_games

    # Create or load agent
    if agent is None:
        # Load checkpoint to get architecture info
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hidden_layers = checkpoint.get("hidden_layers", [256, 256])
        agent = DQNAgent(device=device, hidden_layers=hidden_layers)
        agent.load_checkpoint(checkpoint_path)

    scores: List[int] = []
    games_completed = 0

    while games_completed < num_games:
        state = env.reset()
        episode_scores = torch.zeros(batch_size, device=device)
        done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        while not done_mask.all():
            from game.moves import compute_valid_mask
            valid_mask = compute_valid_mask(state, device)

            # Greedy action selection (no exploration)
            actions = agent.select_action(state, valid_mask, training=False)

            result = env.step(actions)

            # Accumulate scores for active games
            episode_scores += torch.where(
                done_mask,
                torch.zeros_like(result.merge_reward.float()),
                result.merge_reward.float()
            )

            # Track newly completed games (vectorized per DEC-0039)
            newly_done = result.done & ~done_mask
            if newly_done.any():
                new_scores = episode_scores[newly_done]
                remaining_slots = num_games - games_completed
                scores_to_add = new_scores[:remaining_slots].int().tolist()
                scores.extend(scores_to_add)
                games_completed += len(scores_to_add)

            done_mask = done_mask | result.done

            # Update state
            state = torch.where(
                result.done.unsqueeze(-1).unsqueeze(-1),
                result.reset_states,
                result.next_state
            )

            # Sync env internal state
            env._state = state.clone()

    # Trim to exact number requested
    scores = scores[:num_games]

    return EvalResult(
        scores=scores,
        avg_score=sum(scores) / len(scores) if scores else 0.0,
        max_score=max(scores) if scores else 0,
    )


def _create_default_env_factory(n_games: int = 32) -> Callable[[], GameEnv]:
    """Create a default environment factory.

    Args:
        n_games: Number of parallel games

    Returns:
        Factory callable
    """
    def factory() -> GameEnv:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return GameEnv(n_games=n_games, device=device)
    return factory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN Training and Evaluation")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Mode to run")
    parser.add_argument("--config", default="algorithms/dqn/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", help="Checkpoint path for evaluation")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games for evaluation")
    parser.add_argument("--n-envs", type=int, default=32,
                        help="Number of parallel environments")

    args = parser.parse_args()

    env_factory = _create_default_env_factory(args.n_envs)

    if args.mode == "train":
        result = train(env_factory, args.config)
        print(f"\nTraining completed. Checkpoints: {result.checkpoints}")
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint required for evaluation")
        result = evaluate(env_factory, args.checkpoint, args.num_games)
        print(f"\nEvaluation Results:")
        print(f"  Games: {len(result.scores)}")
        print(f"  Avg Score: {result.avg_score:.1f}")
        print(f"  Max Score: {result.max_score}")
