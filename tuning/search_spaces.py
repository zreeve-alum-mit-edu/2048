"""
Hyperparameter Search Spaces.

Defines search spaces for all tunable parameters in DQN training.

Per DEC-0035: epsilon_start fixed at 1.0
Per DEC-0036: Hard target update (frequency is tunable)
Per DEC-0037: Spec packet approved search ranges.
"""

from optuna import Trial
from typing import Dict, Any, List


def suggest_hyperparams(trial: Trial, repr_type: str) -> Dict[str, Any]:
    """Suggest all hyperparameters for a trial.

    Args:
        trial: Optuna trial object
        repr_type: Representation type (onehot, embedding, cnn_2x2, cnn_4x1, cnn_multi)

    Returns:
        Dictionary of hyperparameter values
    """
    params: Dict[str, Any] = {}

    # ===================
    # Training Hyperparameters
    # ===================
    params["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-5, 1e-2, log=True
    )
    params["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256]
    )
    params["gamma"] = trial.suggest_float("gamma", 0.9, 0.999)

    # ===================
    # Epsilon Schedule (DEC-0035)
    # ===================
    params["epsilon_start"] = 1.0  # Fixed per DEC-0035
    params["epsilon_end"] = trial.suggest_float(
        "epsilon_end", 0.001, 0.1, log=True
    )
    params["epsilon_decay_steps"] = trial.suggest_int(
        "epsilon_decay_steps", 50000, 500000, step=50000
    )

    # ===================
    # Target Network (DEC-0036)
    # ===================
    params["target_update_frequency"] = trial.suggest_int(
        "target_update_frequency", 100, 10000, step=100
    )

    # ===================
    # Replay Buffer
    # ===================
    params["buffer_capacity"] = trial.suggest_categorical(
        "buffer_capacity", [50000, 100000, 200000, 500000]
    )
    params["buffer_min_size"] = trial.suggest_int(
        "buffer_min_size", 1000, 10000, step=1000
    )

    # ===================
    # Network Architecture
    # ===================
    n_layers = trial.suggest_int("n_hidden_layers", 1, 3)
    hidden_layers: List[int] = []
    for i in range(n_layers):
        layer_size = trial.suggest_categorical(
            f"hidden_size_{i}", [64, 128, 256, 512]
        )
        hidden_layers.append(layer_size)
    params["hidden_layers"] = hidden_layers

    # ===================
    # Representation-Specific Hyperparameters
    # ===================
    if repr_type == "embedding":
        params["embed_dim"] = trial.suggest_categorical(
            "embed_dim", [8, 16, 32, 64]
        )
    elif repr_type in ("cnn_2x2", "cnn_4x1", "cnn_multi"):
        params["cnn_channels"] = trial.suggest_categorical(
            "cnn_channels", [32, 64, 128]
        )

    return params


def get_default_params(repr_type: str) -> Dict[str, Any]:
    """Get default hyperparameters for testing.

    Args:
        repr_type: Representation type

    Returns:
        Dictionary of default hyperparameter values
    """
    params = {
        "learning_rate": 0.0001,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay_steps": 100000,
        "target_update_frequency": 1000,
        "buffer_capacity": 100000,
        "buffer_min_size": 1000,
        "hidden_layers": [256, 256],
    }

    if repr_type == "embedding":
        params["embed_dim"] = 32
    elif repr_type in ("cnn_2x2", "cnn_4x1", "cnn_multi"):
        params["cnn_channels"] = 64

    return params
