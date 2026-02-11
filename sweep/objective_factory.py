"""
Objective Factory for Algorithm-Specific Optuna Objectives.

Creates objective functions dynamically for each algorithm type,
handling the different agent interfaces and hyperparameter spaces.

Per DEC-0006: Each algorithm implements train(env_factory, config_path).
Per DEC-0012: MedianPruner with epoch-level reporting.
"""

import importlib
import time
import traceback
from typing import Any, Callable, Dict, Optional

import optuna
from optuna import Trial
import torch

from game.env import GameEnv
from game.moves import compute_valid_mask
from sweep.study_factory import SweepStudyConfig
from sweep.observability import SweepObserver


# Algorithm-specific hyperparameter ranges
# These define the search space for each algorithm family

VALUE_BASED_PARAMS = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "gamma": {"type": "float", "low": 0.9, "high": 0.999},
    "epsilon_start": {"type": "fixed", "value": 1.0},
    "epsilon_end": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    "epsilon_decay_steps": {"type": "int", "low": 50000, "high": 500000, "step": 50000},
    "target_update_frequency": {"type": "int", "low": 100, "high": 10000, "step": 100},
    "buffer_capacity": {"type": "categorical", "choices": [50000, 100000, 200000]},
    "buffer_min_size": {"type": "int", "low": 1000, "high": 10000, "step": 1000},
    "n_hidden_layers": {"type": "int", "low": 1, "high": 3},
    "hidden_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
}

POLICY_GRADIENT_PARAMS = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "gamma": {"type": "float", "low": 0.9, "high": 0.999},
    "n_hidden_layers": {"type": "int", "low": 1, "high": 3},
    "hidden_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
}

ACTOR_CRITIC_PARAMS = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "gamma": {"type": "float", "low": 0.9, "high": 0.999},
    "n_steps": {"type": "categorical", "choices": [5, 10, 20, 32, 64]},
    "value_loss_coef": {"type": "float", "low": 0.1, "high": 1.0},
    "entropy_coef": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    "n_hidden_layers": {"type": "int", "low": 1, "high": 3},
    "hidden_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
}

PPO_PARAMS = {
    **ACTOR_CRITIC_PARAMS,
    "clip_ratio": {"type": "float", "low": 0.1, "high": 0.3},
    "gae_lambda": {"type": "float", "low": 0.9, "high": 0.99},
    "n_epochs": {"type": "categorical", "choices": [3, 5, 10]},
    "n_minibatches": {"type": "categorical", "choices": [4, 8, 16]},
}

# Algorithm-specific parameter overrides/additions
ALGORITHM_PARAMS = {
    "dqn": VALUE_BASED_PARAMS,
    "double_dqn": VALUE_BASED_PARAMS,
    "dueling_dqn": VALUE_BASED_PARAMS,
    "per_dqn": {**VALUE_BASED_PARAMS,
                "alpha": {"type": "float", "low": 0.4, "high": 0.8},
                "beta_start": {"type": "float", "low": 0.3, "high": 0.6}},
    "nstep_dqn": {**VALUE_BASED_PARAMS,
                  "n_step": {"type": "categorical", "choices": [3, 5, 10]}},
    "rainbow_lite": {**VALUE_BASED_PARAMS,
                     "n_step": {"type": "categorical", "choices": [3, 5, 10]},
                     "alpha": {"type": "float", "low": 0.4, "high": 0.8},
                     "beta_start": {"type": "float", "low": 0.3, "high": 0.6}},
    "reinforce": POLICY_GRADIENT_PARAMS,
    "a2c": ACTOR_CRITIC_PARAMS,
    "a3c": {**ACTOR_CRITIC_PARAMS,
            "n_workers": {"type": "categorical", "choices": [2, 4, 8]}},
    "ppo_gae": PPO_PARAMS,
    "ppo_value_clip": {**PPO_PARAMS,
                       "value_clip_range": {"type": "float", "low": 0.1, "high": 0.3}},
    "acer": {**ACTOR_CRITIC_PARAMS,
             "replay_ratio": {"type": "categorical", "choices": [1, 2, 4]},
             "c": {"type": "float", "low": 1.0, "high": 20.0}},
    "impala": {**ACTOR_CRITIC_PARAMS,
               "rho_bar": {"type": "float", "low": 0.5, "high": 2.0},
               "c_bar": {"type": "float", "low": 0.5, "high": 2.0}},
    "sarsa": VALUE_BASED_PARAMS,
    "expected_sarsa": VALUE_BASED_PARAMS,
    "sarsa_lambda": {**VALUE_BASED_PARAMS,
                     "lambda_": {"type": "float", "low": 0.5, "high": 0.99}},
    "c51": {**VALUE_BASED_PARAMS,
            "n_atoms": {"type": "categorical", "choices": [21, 51, 101]},
            "v_min": {"type": "float", "low": -100, "high": 0},
            "v_max": {"type": "float", "low": 100, "high": 10000}},
    "qr_dqn": {**VALUE_BASED_PARAMS,
               "n_quantiles": {"type": "categorical", "choices": [32, 64, 128, 200]}},
    "mcts_learned": {**VALUE_BASED_PARAMS,
                     "n_simulations": {"type": "categorical", "choices": [50, 100, 200]},
                     "c_puct": {"type": "float", "low": 0.5, "high": 2.0}},
    "muzero_style": {**VALUE_BASED_PARAMS,
                     "n_simulations": {"type": "categorical", "choices": [50, 100, 200]},
                     "c_puct": {"type": "float", "low": 0.5, "high": 2.0},
                     "unroll_steps": {"type": "categorical", "choices": [3, 5, 10]}},
}

# Representation-specific parameters
REPRESENTATION_PARAMS = {
    "onehot": {},
    "embedding": {"embed_dim": {"type": "categorical", "choices": [8, 16, 32, 64]}},
    "cnn_2x2": {"cnn_channels": {"type": "categorical", "choices": [32, 64, 128]}},
    "cnn_4x1": {"cnn_channels": {"type": "categorical", "choices": [32, 64, 128]}},
    "cnn_multi": {"cnn_channels": {"type": "categorical", "choices": [32, 64, 128]}},
}


def suggest_param(trial: Trial, name: str, spec: Dict[str, Any]) -> Any:
    """Suggest a parameter value based on specification.

    Args:
        trial: Optuna trial
        name: Parameter name
        spec: Parameter specification dict

    Returns:
        Suggested value
    """
    param_type = spec["type"]

    if param_type == "fixed":
        return spec["value"]
    elif param_type == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            log=spec.get("log", False)
        )
    elif param_type == "int":
        return trial.suggest_int(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1)
        )
    elif param_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def suggest_hyperparams(
    trial: Trial,
    algorithm: str,
    representation: str,
) -> Dict[str, Any]:
    """Suggest all hyperparameters for a trial.

    Args:
        trial: Optuna trial
        algorithm: Algorithm name
        representation: Representation type

    Returns:
        Dictionary of hyperparameter values
    """
    params = {}

    # Get algorithm-specific parameters
    algo_params = ALGORITHM_PARAMS.get(algorithm, VALUE_BASED_PARAMS)

    for name, spec in algo_params.items():
        if name == "n_hidden_layers":
            continue  # Handle separately for architecture
        if name == "hidden_size":
            continue  # Handle separately for architecture
        params[name] = suggest_param(trial, name, spec)

    # Handle network architecture
    n_layers = suggest_param(trial, "n_hidden_layers",
                             algo_params.get("n_hidden_layers", {"type": "int", "low": 1, "high": 3}))
    hidden_size_spec = algo_params.get("hidden_size", {"type": "categorical", "choices": [128, 256]})

    hidden_layers = []
    for i in range(n_layers):
        layer_size = trial.suggest_categorical(f"hidden_size_{i}", hidden_size_spec["choices"])
        hidden_layers.append(layer_size)
    params["hidden_layers"] = hidden_layers

    # Get representation-specific parameters
    repr_params = REPRESENTATION_PARAMS.get(representation, {})
    for name, spec in repr_params.items():
        params[name] = suggest_param(trial, name, spec)

    return params


def create_representation(repr_type: str, params: Dict[str, Any]):
    """Create representation instance.

    Args:
        repr_type: Representation type
        params: Hyperparameters

    Returns:
        Representation module
    """
    from representations.onehot import OneHotRepresentation
    from representations.embedding import EmbeddingRepresentation
    from representations.cnn import CNNRepresentation

    if repr_type == "onehot":
        return OneHotRepresentation({})
    elif repr_type == "embedding":
        embed_dim = params.get("embed_dim", 32)
        return EmbeddingRepresentation({"embed_dim": embed_dim})
    elif repr_type == "cnn_2x2":
        channels = params.get("cnn_channels", 64)
        return CNNRepresentation({
            "kernels": [{"size": [2, 2], "out_channels": channels, "stride": [1, 1]}],
            "combine": "concat",
            "activation": "relu"
        })
    elif repr_type == "cnn_4x1":
        channels = params.get("cnn_channels", 64)
        return CNNRepresentation({
            "kernels": [
                {"size": [4, 1], "out_channels": channels, "stride": [1, 1]},
                {"size": [1, 4], "out_channels": channels, "stride": [1, 1]},
            ],
            "combine": "concat",
            "activation": "relu"
        })
    elif repr_type == "cnn_multi":
        channels = params.get("cnn_channels", 64)
        return CNNRepresentation({
            "kernels": [
                {"size": [2, 2], "out_channels": channels, "stride": [1, 1]},
                {"size": [4, 1], "out_channels": channels, "stride": [1, 1]},
                {"size": [1, 4], "out_channels": channels, "stride": [1, 1]},
            ],
            "combine": "concat",
            "activation": "relu"
        })
    else:
        raise ValueError(f"Unknown representation type: {repr_type}")


class ObjectiveFactory:
    """Factory for creating algorithm-specific Optuna objectives."""

    def __init__(self, observer: Optional[SweepObserver] = None):
        """Initialize objective factory.

        Args:
            observer: Optional sweep observer for logging
        """
        self.observer = observer

    def create_objective(
        self,
        config: SweepStudyConfig,
    ) -> Callable[[Trial], float]:
        """Create objective function for a study.

        Args:
            config: Study configuration

        Returns:
            Objective function that takes a Trial and returns float (avg score)
        """
        algorithm = config.algorithm
        representation = config.representation
        reward_type = config.reward_type
        epochs_per_trial = config.epochs_per_trial
        steps_per_epoch = config.steps_per_epoch
        eval_games = config.eval_games_per_epoch
        observer = self.observer

        def objective(trial: Trial) -> float:
            """Train algorithm and return final evaluation score."""
            trial_start = time.time()

            if observer:
                observer.trial_started(trial.number)

            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Suggest hyperparameters
                params = suggest_hyperparams(trial, algorithm, representation)

                # Create representation
                repr_module = create_representation(representation, params)
                repr_module = repr_module.to(device)

                # Create agent based on algorithm type
                agent = _create_agent(algorithm, device, repr_module, params)

                # Create environment
                env = GameEnv(n_games=32, device=device)

                # Training loop with epoch-level reporting
                state = env.reset()
                total_reward = torch.zeros(env.n_games, device=device)

                for epoch in range(epochs_per_trial):
                    epoch_scores = []

                    for step in range(steps_per_epoch):
                        valid_mask = compute_valid_mask(state, device)
                        actions = agent.select_action(state, valid_mask, training=True)

                        result = env.step(actions)

                        # Get reward based on reward type
                        if reward_type == "merge":
                            reward = result.merge_reward.float()
                        else:
                            reward = result.spawn_reward.float()

                        # Store transition (algorithm-specific handling)
                        _store_transition(agent, algorithm, state, actions, reward,
                                          result, device)

                        total_reward += reward

                        if result.done.any():
                            done_scores = total_reward[result.done]
                            epoch_scores.extend(done_scores.tolist())
                            total_reward = torch.where(result.done,
                                                       torch.zeros_like(total_reward),
                                                       total_reward)

                        # Train step (algorithm-specific)
                        _train_step(agent, algorithm, state, result, device)

                        # Update state
                        state = torch.where(
                            result.done.unsqueeze(-1).unsqueeze(-1),
                            result.reset_states,
                            result.next_state
                        )
                        env._state = state.clone()

                    # Evaluate at end of epoch
                    eval_score = _quick_eval(agent, algorithm, device, eval_games)

                    # Report to Optuna for pruning
                    trial.report(eval_score, epoch)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                # Final evaluation
                final_score = _quick_eval(agent, algorithm, device, num_games=100)

                duration = time.time() - trial_start
                if observer:
                    observer.trial_completed(
                        trial_number=trial.number,
                        score=final_score,
                        duration_seconds=duration,
                        params=params,
                        pruned=False,
                    )

                return final_score

            except optuna.TrialPruned:
                duration = time.time() - trial_start
                if observer:
                    observer.trial_completed(
                        trial_number=trial.number,
                        score=0.0,
                        duration_seconds=duration,
                        params={},
                        pruned=True,
                    )
                raise

            except Exception as e:
                duration = time.time() - trial_start
                # Log error but let Optuna handle it
                print(f"  [ERROR] Trial {trial.number} failed: {e}")
                raise

        return objective


def _create_agent(algorithm: str, device: torch.device, repr_module, params: Dict[str, Any]):
    """Create agent for specific algorithm.

    Args:
        algorithm: Algorithm name
        device: PyTorch device
        repr_module: Representation module
        params: Hyperparameters

    Returns:
        Agent instance
    """
    # Import algorithm-specific agent
    if algorithm == "dqn":
        from algorithms.dqn.agent import DQNAgent
        return DQNAgent(
            device=device,
            representation=repr_module,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
        )

    elif algorithm == "double_dqn":
        from algorithms.double_dqn.agent import DoubleDQNAgent
        return DoubleDQNAgent(
            device=device,
            representation=repr_module,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
        )

    elif algorithm == "dueling_dqn":
        from algorithms.dueling_dqn.agent import DuelingDQNAgent
        return DuelingDQNAgent(
            device=device,
            representation=repr_module,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
        )

    elif algorithm == "per_dqn":
        from algorithms.per_dqn.agent import PERDQNAgent
        return PERDQNAgent(
            device=device,
            representation=repr_module,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
            alpha=params.get("alpha", 0.6),
            beta_start=params.get("beta_start", 0.4),
        )

    elif algorithm == "nstep_dqn":
        from algorithms.nstep_dqn.agent import NStepDQNAgent
        return NStepDQNAgent(
            device=device,
            representation=repr_module,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
            n_step=params.get("n_step", 5),
        )

    elif algorithm == "rainbow_lite":
        from algorithms.rainbow_lite.agent import RainbowLiteAgent
        return RainbowLiteAgent(
            device=device,
            representation=repr_module,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
            n_step=params.get("n_step", 5),
            alpha=params.get("alpha", 0.6),
            beta_start=params.get("beta_start", 0.4),
        )

    elif algorithm == "reinforce":
        from algorithms.reinforce.agent import REINFORCEAgent
        return REINFORCEAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
        )

    elif algorithm == "a2c":
        from algorithms.a2c.agent import A2CAgent
        return A2CAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_steps=params.get("n_steps", 5),
            value_loss_coef=params.get("value_loss_coef", 0.5),
            entropy_coef=params.get("entropy_coef", 0.01),
        )

    elif algorithm == "a3c":
        from algorithms.a3c.agent import A3CAgent
        return A3CAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_steps=params.get("n_steps", 5),
            value_loss_coef=params.get("value_loss_coef", 0.5),
            entropy_coef=params.get("entropy_coef", 0.01),
        )

    elif algorithm == "ppo_gae":
        from algorithms.ppo_gae.agent import PPOAgent
        return PPOAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            gae_lambda=params.get("gae_lambda", 0.95),
            clip_ratio=params.get("clip_ratio", 0.2),
            n_steps=params.get("n_steps", 32),
            n_epochs=params.get("n_epochs", 5),
            n_minibatches=params.get("n_minibatches", 4),
            value_loss_coef=params.get("value_loss_coef", 0.5),
            entropy_coef=params.get("entropy_coef", 0.01),
        )

    elif algorithm == "ppo_value_clip":
        from algorithms.ppo_value_clip.agent import PPOValueClipAgent
        return PPOValueClipAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            gae_lambda=params.get("gae_lambda", 0.95),
            clip_ratio=params.get("clip_ratio", 0.2),
            value_clip_range=params.get("value_clip_range", 0.2),
            n_steps=params.get("n_steps", 32),
            n_epochs=params.get("n_epochs", 5),
            n_minibatches=params.get("n_minibatches", 4),
            value_loss_coef=params.get("value_loss_coef", 0.5),
            entropy_coef=params.get("entropy_coef", 0.01),
        )

    elif algorithm == "acer":
        from algorithms.acer.agent import ACERAgent
        return ACERAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_steps=params.get("n_steps", 5),
            value_loss_coef=params.get("value_loss_coef", 0.5),
            entropy_coef=params.get("entropy_coef", 0.01),
            replay_ratio=params.get("replay_ratio", 4),
            c=params.get("c", 10.0),
        )

    elif algorithm == "impala":
        from algorithms.impala.agent import IMPALAAgent
        return IMPALAAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_steps=params.get("n_steps", 5),
            value_loss_coef=params.get("value_loss_coef", 0.5),
            entropy_coef=params.get("entropy_coef", 0.01),
            rho_bar=params.get("rho_bar", 1.0),
            c_bar=params.get("c_bar", 1.0),
        )

    elif algorithm == "sarsa":
        from algorithms.sarsa.agent import SARSAAgent
        return SARSAAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
        )

    elif algorithm == "expected_sarsa":
        from algorithms.expected_sarsa.agent import ExpectedSARSAAgent
        return ExpectedSARSAAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
        )

    elif algorithm == "sarsa_lambda":
        from algorithms.sarsa_lambda.agent import SARSALambdaAgent
        return SARSALambdaAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            lambda_=params.get("lambda_", 0.9),
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
        )

    elif algorithm == "c51":
        from algorithms.c51.agent import C51Agent
        return C51Agent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
            n_atoms=params.get("n_atoms", 51),
            v_min=params.get("v_min", -10),
            v_max=params.get("v_max", 5000),
        )

    elif algorithm == "qr_dqn":
        from algorithms.qr_dqn.agent import QRDQNAgent
        return QRDQNAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            epsilon_start=params.get("epsilon_start", 1.0),
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            target_update_frequency=params["target_update_frequency"],
            buffer_capacity=params["buffer_capacity"],
            buffer_min_size=params["buffer_min_size"],
            batch_size=params["batch_size"],
            n_quantiles=params.get("n_quantiles", 64),
        )

    elif algorithm == "mcts_learned":
        from algorithms.mcts_learned.agent import MCTSLearnedAgent
        return MCTSLearnedAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_simulations=params.get("n_simulations", 100),
            c_puct=params.get("c_puct", 1.0),
        )

    elif algorithm == "muzero_style":
        from algorithms.muzero_style.agent import MuZeroStyleAgent
        return MuZeroStyleAgent(
            device=device,
            hidden_layers=params["hidden_layers"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_simulations=params.get("n_simulations", 50),
            c_puct=params.get("c_puct", 1.0),
            unroll_steps=params.get("unroll_steps", 5),
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _store_transition(agent, algorithm: str, state, actions, reward, result, device):
    """Store transition for algorithm-specific agent.

    Different agents have different transition storage interfaces.
    """
    # Handle valid mask for next state
    if result.done.any():
        reset_valid_mask = compute_valid_mask(result.reset_states, device)
        next_valid_mask = torch.where(
            result.done.unsqueeze(-1),
            reset_valid_mask,
            result.valid_mask
        )
    else:
        next_valid_mask = result.valid_mask

    # Value-based algorithms with replay buffers
    if algorithm in ["dqn", "double_dqn", "dueling_dqn", "per_dqn", "nstep_dqn",
                     "rainbow_lite", "c51", "qr_dqn"]:
        agent.store_transition(
            state=state,
            action=actions,
            reward=reward,
            next_state=result.next_state,
            done=result.done,
            valid_mask=next_valid_mask,
        )

    # Policy gradient / actor-critic algorithms
    elif algorithm in ["reinforce", "a2c", "a3c", "acer", "impala"]:
        # These store during select_action or need different interface
        pass

    # PPO stores during rollout collection
    elif algorithm in ["ppo_gae", "ppo_value_clip"]:
        pass

    # SARSA variants
    elif algorithm in ["sarsa", "expected_sarsa", "sarsa_lambda"]:
        # Online learning, handled in train_step
        pass

    # Model-based
    elif algorithm in ["mcts_learned", "muzero_style"]:
        pass


def _train_step(agent, algorithm: str, state, result, device):
    """Perform training step for algorithm-specific agent."""

    # Value-based with standard train_step
    if algorithm in ["dqn", "double_dqn", "dueling_dqn", "per_dqn", "nstep_dqn",
                     "rainbow_lite", "c51", "qr_dqn"]:
        agent.train_step()

    # Actor-critic with next state bootstrap
    elif algorithm in ["a2c", "a3c", "acer", "impala"]:
        # These need periodic training at end of rollout
        pass

    # PPO trains at end of rollout
    elif algorithm in ["ppo_gae", "ppo_value_clip"]:
        pass

    # REINFORCE trains at end of episode
    elif algorithm == "reinforce":
        pass

    # SARSA variants train online
    elif algorithm in ["sarsa", "expected_sarsa", "sarsa_lambda"]:
        pass

    # Model-based
    elif algorithm in ["mcts_learned", "muzero_style"]:
        agent.train_step()


def _quick_eval(agent, algorithm: str, device: torch.device, num_games: int) -> float:
    """Quick evaluation of agent.

    Args:
        agent: Agent to evaluate
        algorithm: Algorithm name
        device: PyTorch device
        num_games: Number of games

    Returns:
        Average score
    """
    env = GameEnv(n_games=32, device=device)
    scores = []
    games_completed = 0

    while games_completed < num_games:
        state = env.reset()
        episode_scores = torch.zeros(32, device=device)
        done_mask = torch.zeros(32, dtype=torch.bool, device=device)

        while not done_mask.all():
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
                remaining = num_games - games_completed
                scores_to_add = new_scores[:remaining].tolist()
                scores.extend(scores_to_add)
                games_completed += len(scores_to_add)

            done_mask = done_mask | result.done

            state = torch.where(
                result.done.unsqueeze(-1).unsqueeze(-1),
                result.reset_states,
                result.next_state
            )
            env._state = state.clone()

    return sum(scores[:num_games]) / num_games if scores else 0.0
