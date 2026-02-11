"""
MuZero-style Entry Point.

Per DEC-0005: Algorithm modules MUST be self-contained in algorithms/<name>/
Per DEC-0006: run.py MUST implement train() and evaluate() with specified signatures

Milestone 24: MuZero-style model-based planning.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional

import torch
import yaml

from algorithms.muzero_style.agent import MuZeroAgent
from algorithms.muzero_style.buffer import Trajectory
from game.env import GameEnv
from game.moves import compute_valid_mask


@dataclass
class TrainingResult:
    """Result of training run. Per DEC-0006."""
    checkpoints: List[str]
    metrics: Dict[str, Any]


@dataclass
class EvalResult:
    """Result of evaluation run. Per DEC-0006."""
    scores: List[int]
    avg_score: float
    max_score: int


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(
    env_factory: Callable[[], GameEnv],
    config_path: str
) -> TrainingResult:
    """Train MuZero-style agent.

    Training loop:
    1. Play games with MCTS using learned model
    2. Store complete trajectories
    3. Train networks on sampled trajectory positions

    Per DEC-0006: Required interface for algorithm training.
    """
    config = load_config(config_path)

    # Create single-game environment for self-play
    env = GameEnv(n_games=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = env.device
    print(f"Using device: {device}")

    agent = MuZeroAgent(
        device=device,
        hidden_size=config["network"]["hidden_size"],
        representation_layers=config["network"]["representation_layers"],
        dynamics_layers=config["network"]["dynamics_layers"],
        prediction_layers=config["network"]["prediction_layers"],
        learning_rate=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        num_simulations=config["mcts"]["num_simulations"],
        c_puct=config["mcts"]["c_puct"],
        dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
        exploration_fraction=config["mcts"]["exploration_fraction"],
        temperature=config["mcts"]["temperature"],
        temperature_drop_step=config["mcts"]["temperature_drop_step"],
        buffer_capacity=config["replay_buffer"]["capacity"],
        td_steps=config["training"]["td_steps"],
        unroll_steps=config["training"]["unroll_steps"],
        batch_size=config["training"]["batch_size"],
        value_loss_weight=config["training"]["value_loss_weight"],
        policy_loss_weight=config["training"]["policy_loss_weight"],
        reward_loss_weight=config["training"]["reward_loss_weight"],
    )

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
        "policy_losses": [],
        "value_losses": [],
        "reward_losses": [],
        "eval_scores": [],
        "episode_scores": [],
    }

    completed_episodes = 0
    total_episode_score = 0.0
    step = 0

    print(f"Starting MuZero-style training for {total_steps} steps...")
    print(f"MCTS simulations: {config['mcts']['num_simulations']}")
    print(f"Unroll steps: {config['training']['unroll_steps']}")

    while step < total_steps:
        # Play one game with MCTS using learned model
        state = env.reset().squeeze(0)  # (16, 17)

        observations = [state.clone()]
        actions = []
        rewards = []
        policies = []
        values = []
        valid_masks = []

        game_done = False
        episode_score = 0.0

        while not game_done:
            valid_mask = compute_valid_mask(
                state.unsqueeze(0), device
            ).squeeze(0)
            valid_masks.append(valid_mask.clone())

            # MCTS action selection using learned model
            action, policy_target, value_est = agent.select_action(
                state, valid_mask, training=True
            )

            policies.append(policy_target.clone())
            values.append(value_est)
            actions.append(action)

            # Take action in real environment
            action_tensor = torch.tensor([action], device=device)
            env._state = state.unsqueeze(0)
            result = env.step(action_tensor)

            reward = result.merge_reward.squeeze(0).item()
            rewards.append(reward)
            episode_score += reward

            game_done = result.done.squeeze(0).item()

            if game_done:
                state = result.reset_states.squeeze(0)
            else:
                state = result.next_state.squeeze(0)

            observations.append(state.clone())

        # Add final valid mask
        final_valid_mask = compute_valid_mask(
            state.unsqueeze(0), device
        ).squeeze(0)
        valid_masks.append(final_valid_mask.clone())

        # Compute value targets (n-step returns handled by buffer)
        # For now, use discounted returns as initial value targets
        T = len(rewards)
        value_targets = []
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + agent.gamma * G
            value_targets.insert(0, G)

        # Store trajectory
        trajectory = Trajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            policies=policies,
            values=value_targets,
            valid_masks=valid_masks,
        )
        agent.store_trajectory(trajectory)

        # Train
        train_metrics = agent.train_step()

        if train_metrics is not None:
            metrics["losses"].append(train_metrics["loss"])
            metrics["policy_losses"].append(train_metrics["policy_loss"])
            metrics["value_losses"].append(train_metrics["value_loss"])
            metrics["reward_losses"].append(train_metrics["reward_loss"])

        # Update counters
        completed_episodes += 1
        total_episode_score += episode_score
        metrics["episode_scores"].append(episode_score)
        step += T

        # Logging
        if step >= log_frequency and step % log_frequency < T:
            avg_loss = sum(metrics["losses"][-100:]) / len(metrics["losses"][-100:]) if metrics["losses"] else 0
            avg_policy = sum(metrics["policy_losses"][-100:]) / len(metrics["policy_losses"][-100:]) if metrics["policy_losses"] else 0
            avg_value = sum(metrics["value_losses"][-100:]) / len(metrics["value_losses"][-100:]) if metrics["value_losses"] else 0
            avg_reward = sum(metrics["reward_losses"][-100:]) / len(metrics["reward_losses"][-100:]) if metrics["reward_losses"] else 0
            avg_ep_score = total_episode_score / completed_episodes if completed_episodes > 0 else 0

            print(f"Step {step}/{total_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"P: {avg_policy:.4f} | "
                  f"V: {avg_value:.4f} | "
                  f"R: {avg_reward:.4f} | "
                  f"Ep: {completed_episodes} | "
                  f"Score: {avg_ep_score:.1f}")

        # Evaluation
        if step >= eval_frequency and step % eval_frequency < T:
            eval_result = evaluate(env_factory, None, eval_games, agent=agent)
            metrics["eval_scores"].append({
                "step": step,
                "avg_score": eval_result.avg_score,
                "max_score": eval_result.max_score,
            })
            print(f"  Eval: Avg={eval_result.avg_score:.1f}, Max={eval_result.max_score}")

        # Checkpointing
        if step >= checkpoint_frequency and step % checkpoint_frequency < T:
            checkpoint_path = str(checkpoint_dir / f"muzero_style_step_{step}.pt")
            agent.save_checkpoint(checkpoint_path)
            checkpoints.append(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    final_checkpoint = str(checkpoint_dir / "muzero_style_final.pt")
    agent.save_checkpoint(final_checkpoint)
    checkpoints.append(final_checkpoint)

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
    agent: Optional[MuZeroAgent] = None
) -> EvalResult:
    """Evaluate MuZero-style agent.

    Uses deterministic MCTS for evaluation.

    Per DEC-0006.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GameEnv(n_games=1, device=device)

    if agent is None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hidden_size = checkpoint.get("hidden_size", 256)
        mcts_cfg = checkpoint.get("mcts_config", {})
        agent = MuZeroAgent(
            device=device,
            hidden_size=hidden_size,
            num_simulations=mcts_cfg.get("num_simulations", 50),
            c_puct=mcts_cfg.get("c_puct", 1.5),
        )
        agent.load_checkpoint(checkpoint_path)

    scores: List[int] = []

    for game_idx in range(num_games):
        state = env.reset().squeeze(0)
        episode_score = 0.0
        game_done = False

        while not game_done:
            valid_mask = compute_valid_mask(
                state.unsqueeze(0), device
            ).squeeze(0)

            # Deterministic MCTS
            action, _, _ = agent.select_action(state, valid_mask, training=False)

            action_tensor = torch.tensor([action], device=device)
            env._state = state.unsqueeze(0)
            result = env.step(action_tensor)

            episode_score += result.merge_reward.squeeze(0).item()
            game_done = result.done.squeeze(0).item()

            if game_done:
                state = result.reset_states.squeeze(0)
            else:
                state = result.next_state.squeeze(0)

        scores.append(int(episode_score))

        if (game_idx + 1) % 10 == 0:
            print(f"  Evaluated {game_idx + 1}/{num_games} games...")

    return EvalResult(
        scores=scores,
        avg_score=sum(scores) / len(scores) if scores else 0.0,
        max_score=max(scores) if scores else 0,
    )


def _create_default_env_factory(n_games: int = 1) -> Callable[[], GameEnv]:
    """Create a default environment factory."""
    def factory() -> GameEnv:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return GameEnv(n_games=n_games, device=device)
    return factory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuZero-style Training and Evaluation")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Mode to run")
    parser.add_argument("--config", default="algorithms/muzero_style/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", help="Checkpoint path for evaluation")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games for evaluation")

    args = parser.parse_args()
    env_factory = _create_default_env_factory()

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
