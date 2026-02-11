"""
MCTS+Learned Entry Point.

Per DEC-0005: Algorithm modules MUST be self-contained in algorithms/<name>/
Per DEC-0006: run.py MUST implement train() and evaluate() with specified signatures

Milestone 25: MCTS with learned value/policy networks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional

import torch
import yaml

from algorithms.mcts_learned.agent import MCTSLearnedAgent
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
    """Train MCTS+Learned agent.

    Training loop:
    1. Play games with MCTS action selection
    2. Store (state, MCTS_policy, return) tuples
    3. Train network on replay buffer

    Per DEC-0006: Required interface for algorithm training.
    """
    config = load_config(config_path)

    # Create environment (use single game for MCTS self-play)
    env = GameEnv(n_games=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = env.device
    print(f"Using device: {device}")

    agent = MCTSLearnedAgent(
        device=device,
        hidden_layers=config["network"]["hidden_layers"],
        learning_rate=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        num_simulations=config["mcts"]["num_simulations"],
        c_puct=config["mcts"]["c_puct"],
        dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
        exploration_fraction=config["mcts"]["exploration_fraction"],
        temperature=config["mcts"]["temperature"],
        temperature_drop_step=config["mcts"]["temperature_drop_step"],
        buffer_capacity=config["replay_buffer"]["capacity"],
        buffer_min_size=config["replay_buffer"]["min_size"],
        batch_size=config["training"]["batch_size"],
        value_loss_weight=config["training"]["value_loss_weight"],
        policy_loss_weight=config["training"]["policy_loss_weight"],
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
        "eval_scores": [],
        "episode_scores": [],
    }

    # Self-play training
    completed_episodes = 0
    total_episode_score = 0.0
    step = 0

    print(f"Starting MCTS+Learned training for {total_steps} steps...")
    print(f"MCTS simulations: {config['mcts']['num_simulations']}")

    while step < total_steps:
        # Play one game with MCTS
        state = env.reset().squeeze(0)  # (16, 17)
        episode_states = []
        episode_policies = []
        episode_rewards = []
        episode_valid_masks = []

        game_done = False
        episode_score = 0.0

        while not game_done:
            valid_mask = compute_valid_mask(
                state.unsqueeze(0), device
            ).squeeze(0)

            # MCTS action selection
            action, policy_target = agent.select_action(
                state, valid_mask, training=True
            )

            # Store state info
            episode_states.append(state.clone())
            episode_policies.append(policy_target.clone())
            episode_valid_masks.append(valid_mask.clone())

            # Take action
            action_tensor = torch.tensor([action], device=device)
            env._state = state.unsqueeze(0)
            result = env.step(action_tensor)

            reward = result.merge_reward.squeeze(0).item()
            episode_rewards.append(reward)
            episode_score += reward

            game_done = result.done.squeeze(0).item()

            if game_done:
                state = result.reset_states.squeeze(0)
            else:
                state = result.next_state.squeeze(0)

        # Compute value targets (discounted returns)
        T = len(episode_rewards)
        value_targets = []
        G = 0.0
        for t in reversed(range(T)):
            G = episode_rewards[t] + agent.gamma * G
            value_targets.insert(0, G)

        # Store experiences
        for t in range(T):
            agent.store_experience(
                episode_states[t],
                episode_policies[t],
                value_targets[t],
                episode_valid_masks[t],
            )

        # Train
        train_metrics = agent.train_step()

        if train_metrics is not None:
            metrics["losses"].append(train_metrics["loss"])
            metrics["policy_losses"].append(train_metrics["policy_loss"])
            metrics["value_losses"].append(train_metrics["value_loss"])

        # Update counters
        completed_episodes += 1
        total_episode_score += episode_score
        metrics["episode_scores"].append(episode_score)
        step += T  # Count each move as a step

        # Logging
        if step >= log_frequency and step % log_frequency < T:
            avg_loss = sum(metrics["losses"][-100:]) / len(metrics["losses"][-100:]) if metrics["losses"] else 0
            avg_policy_loss = sum(metrics["policy_losses"][-100:]) / len(metrics["policy_losses"][-100:]) if metrics["policy_losses"] else 0
            avg_value_loss = sum(metrics["value_losses"][-100:]) / len(metrics["value_losses"][-100:]) if metrics["value_losses"] else 0
            avg_ep_score = total_episode_score / completed_episodes if completed_episodes > 0 else 0

            print(f"Step {step}/{total_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Policy: {avg_policy_loss:.4f} | "
                  f"Value: {avg_value_loss:.4f} | "
                  f"Episodes: {completed_episodes} | "
                  f"Avg Score: {avg_ep_score:.1f}")

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
            checkpoint_path = str(checkpoint_dir / f"mcts_learned_step_{step}.pt")
            agent.save_checkpoint(checkpoint_path)
            checkpoints.append(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    final_checkpoint = str(checkpoint_dir / "mcts_learned_final.pt")
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
    agent: Optional[MCTSLearnedAgent] = None
) -> EvalResult:
    """Evaluate MCTS+Learned agent.

    Uses deterministic MCTS (temperature=0) for evaluation.

    Per DEC-0006.
    """
    # Use single-game env for MCTS evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GameEnv(n_games=1, device=device)

    if agent is None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hidden_layers = checkpoint.get("hidden_layers", [256, 256])
        mcts_cfg = checkpoint.get("mcts_config", {})
        agent = MCTSLearnedAgent(
            device=device,
            hidden_layers=hidden_layers,
            num_simulations=mcts_cfg.get("num_simulations", 100),
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

            # Deterministic MCTS (no exploration)
            action, _ = agent.select_action(state, valid_mask, training=False)

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

    parser = argparse.ArgumentParser(description="MCTS+Learned Training and Evaluation")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Mode to run")
    parser.add_argument("--config", default="algorithms/mcts_learned/config.yaml",
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
