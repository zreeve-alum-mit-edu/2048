"""
PPO with Value Clipping Entry Point.

Per DEC-0005: Algorithm modules MUST be self-contained in algorithms/<name>/
Per DEC-0006: run.py MUST implement train() and evaluate() with specified signatures

Milestone 11: PPO variant with value function clipping.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional

import torch
import yaml

from algorithms.ppo_value_clip.agent import PPOValueClipAgent
from game.env import GameEnv


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
    """Train PPO+Value Clip agent.

    Per DEC-0006: Required interface for algorithm training.
    """
    config = load_config(config_path)

    env = env_factory()
    device = env.device
    print(f"Using device: {device}")

    agent = PPOValueClipAgent(
        device=device,
        hidden_layers=config["network"]["hidden_layers"],
        learning_rate=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_ratio=config["ppo"]["clip_ratio"],
        value_clip_ratio=config["ppo"]["value_clip_ratio"],
        n_steps=config["ppo"]["n_steps"],
        n_epochs=config["ppo"]["n_epochs"],
        n_minibatches=config["ppo"]["n_minibatches"],
        value_loss_coef=config["training"]["value_loss_coef"],
        entropy_coef=config["training"]["entropy_coef"],
    )

    total_steps = config["training"]["total_steps"]
    n_steps = config["ppo"]["n_steps"]
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
        "entropies": [],
        "eval_scores": [],
        "episode_scores": [],
    }

    state = env.reset()
    episode_scores = torch.zeros(env.n_games, device=device)
    completed_episodes = 0
    total_episode_score = 0.0

    print(f"Starting PPO+Value Clip training for {total_steps} steps...")

    step = 0
    while step < total_steps:
        for _ in range(n_steps):
            from game.moves import compute_valid_mask
            valid_mask = compute_valid_mask(state, device)

            actions, log_probs, values = agent.select_action_with_log_prob(state, valid_mask)
            result = env.step(actions)

            agent.store_transition(
                state=state,
                action=actions,
                reward=result.merge_reward.float(),
                done=result.done,
                valid_mask=valid_mask,
                log_prob=log_probs,
                value=values,
            )

            episode_scores += result.merge_reward.float()

            if result.done.any():
                done_scores = episode_scores[result.done]
                completed_episodes += done_scores.numel()
                total_episode_score += done_scores.sum().item()
                metrics["episode_scores"].extend(done_scores.tolist())
                episode_scores = torch.where(result.done, torch.zeros_like(episode_scores), episode_scores)

            state = torch.where(
                result.done.unsqueeze(-1).unsqueeze(-1),
                result.reset_states,
                result.next_state
            )
            env._state = state.clone()

            step += 1
            if step >= total_steps:
                break

        current_done = torch.zeros(env.n_games, dtype=torch.bool, device=device)
        train_metrics = agent.train_step(next_state=state, next_done=current_done)

        if train_metrics is not None:
            metrics["losses"].append(train_metrics["loss"])
            metrics["policy_losses"].append(train_metrics["policy_loss"])
            metrics["value_losses"].append(train_metrics["value_loss"])
            metrics["entropies"].append(train_metrics["entropy"])

        if step % log_frequency == 0:
            avg_loss = sum(metrics["losses"][-100:]) / len(metrics["losses"][-100:]) if metrics["losses"] else 0
            avg_ep_score = total_episode_score / completed_episodes if completed_episodes > 0 else 0
            print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | "
                  f"Episodes: {completed_episodes} | Avg Score: {avg_ep_score:.1f}")

        if step % eval_frequency == 0:
            eval_result = evaluate(env_factory, None, eval_games, agent=agent)
            metrics["eval_scores"].append({
                "step": step,
                "avg_score": eval_result.avg_score,
                "max_score": eval_result.max_score,
            })
            print(f"  Eval: Avg={eval_result.avg_score:.1f}, Max={eval_result.max_score}")

        if step % checkpoint_frequency == 0:
            checkpoint_path = str(checkpoint_dir / f"ppo_value_clip_step_{step}.pt")
            agent.save_checkpoint(checkpoint_path)
            checkpoints.append(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    final_checkpoint = str(checkpoint_dir / "ppo_value_clip_final.pt")
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
    agent: Optional[PPOValueClipAgent] = None
) -> EvalResult:
    """Evaluate PPO+Value Clip agent. Per DEC-0006."""
    env = env_factory()
    device = env.device
    batch_size = env.n_games

    if agent is None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hidden_layers = checkpoint.get("hidden_layers", [256, 256])
        agent = PPOValueClipAgent(device=device, hidden_layers=hidden_layers)
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
            actions = agent.select_action(state, valid_mask, training=False)
            result = env.step(actions)

            episode_scores += torch.where(
                done_mask,
                torch.zeros_like(result.merge_reward.float()),
                result.merge_reward.float()
            )

            newly_done = result.done & ~done_mask
            if newly_done.any():
                new_scores = episode_scores[newly_done]
                remaining_slots = num_games - games_completed
                scores_to_add = new_scores[:remaining_slots].int().tolist()
                scores.extend(scores_to_add)
                games_completed += len(scores_to_add)

            done_mask = done_mask | result.done
            state = torch.where(
                result.done.unsqueeze(-1).unsqueeze(-1),
                result.reset_states,
                result.next_state
            )
            env._state = state.clone()

    scores = scores[:num_games]

    return EvalResult(
        scores=scores,
        avg_score=sum(scores) / len(scores) if scores else 0.0,
        max_score=max(scores) if scores else 0,
    )


def _create_default_env_factory(n_games: int = 32) -> Callable[[], GameEnv]:
    """Create a default environment factory."""
    def factory() -> GameEnv:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return GameEnv(n_games=n_games, device=device)
    return factory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO+Value Clip Training and Evaluation")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Mode to run")
    parser.add_argument("--config", default="algorithms/ppo_value_clip/config.yaml",
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
