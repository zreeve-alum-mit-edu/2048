"""
Tests for Objective Factory.

Tests the creation of algorithm-specific objectives.
"""

import pytest
import optuna

from sweep.objective_factory import (
    ObjectiveFactory,
    suggest_param,
    suggest_hyperparams,
    create_representation,
    ALGORITHM_PARAMS,
    REPRESENTATION_PARAMS,
)
from sweep.study_factory import SweepStudyConfig


class TestSuggestParam:
    """Tests for suggest_param function."""

    def test_fixed_param(self):
        """Fixed parameters return specified value."""
        trial = optuna.create_study().ask()
        value = suggest_param(trial, "test", {"type": "fixed", "value": 1.0})
        assert value == 1.0

    def test_float_param(self):
        """Float parameters are within range."""
        trial = optuna.create_study().ask()
        value = suggest_param(trial, "test", {"type": "float", "low": 0.1, "high": 0.9})
        assert 0.1 <= value <= 0.9

    def test_int_param(self):
        """Int parameters are within range."""
        trial = optuna.create_study().ask()
        value = suggest_param(trial, "test", {"type": "int", "low": 1, "high": 10})
        assert 1 <= value <= 10
        assert isinstance(value, int)

    def test_categorical_param(self):
        """Categorical parameters are from choices."""
        trial = optuna.create_study().ask()
        value = suggest_param(trial, "test", {"type": "categorical", "choices": [32, 64, 128]})
        assert value in [32, 64, 128]


class TestSuggestHyperparams:
    """Tests for suggest_hyperparams function."""

    def test_returns_dict(self):
        """suggest_hyperparams returns dictionary."""
        trial = optuna.create_study().ask()
        params = suggest_hyperparams(trial, "dqn", "onehot")
        assert isinstance(params, dict)

    def test_dqn_params(self):
        """DQN gets value-based parameters."""
        trial = optuna.create_study().ask()
        params = suggest_hyperparams(trial, "dqn", "onehot")

        # Check required DQN params present
        assert "learning_rate" in params
        assert "batch_size" in params
        assert "gamma" in params
        assert "epsilon_end" in params
        assert "target_update_frequency" in params
        assert "buffer_capacity" in params
        assert "hidden_layers" in params

    def test_reinforce_params(self):
        """REINFORCE gets policy gradient parameters."""
        trial = optuna.create_study().ask()
        params = suggest_hyperparams(trial, "reinforce", "onehot")

        assert "learning_rate" in params
        assert "gamma" in params
        assert "hidden_layers" in params

    def test_a2c_params(self):
        """A2C gets actor-critic parameters."""
        trial = optuna.create_study().ask()
        params = suggest_hyperparams(trial, "a2c", "onehot")

        assert "learning_rate" in params
        assert "gamma" in params
        assert "value_loss_coef" in params
        assert "entropy_coef" in params

    def test_ppo_params(self):
        """PPO gets PPO-specific parameters."""
        trial = optuna.create_study().ask()
        params = suggest_hyperparams(trial, "ppo_gae", "onehot")

        assert "clip_ratio" in params
        assert "gae_lambda" in params

    def test_embedding_params(self):
        """Embedding representation adds embed_dim."""
        trial = optuna.create_study().ask()
        params = suggest_hyperparams(trial, "dqn", "embedding")

        assert "embed_dim" in params
        assert params["embed_dim"] in [8, 16, 32, 64]

    def test_cnn_params(self):
        """CNN representation adds cnn_channels."""
        trial = optuna.create_study().ask()
        params = suggest_hyperparams(trial, "dqn", "cnn_2x2")

        assert "cnn_channels" in params
        assert params["cnn_channels"] in [32, 64, 128]

    def test_hidden_layers_is_list(self):
        """hidden_layers is always a list of ints."""
        trial = optuna.create_study().ask()
        params = suggest_hyperparams(trial, "dqn", "onehot")

        assert isinstance(params["hidden_layers"], list)
        assert len(params["hidden_layers"]) >= 1
        assert all(isinstance(x, int) for x in params["hidden_layers"])


class TestCreateRepresentation:
    """Tests for create_representation function."""

    def test_onehot_creation(self):
        """Creates onehot representation."""
        repr_module = create_representation("onehot", {})
        assert repr_module is not None

    def test_embedding_creation(self):
        """Creates embedding representation."""
        repr_module = create_representation("embedding", {"embed_dim": 32})
        assert repr_module is not None

    def test_cnn_2x2_creation(self):
        """Creates CNN 2x2 representation."""
        repr_module = create_representation("cnn_2x2", {"cnn_channels": 64})
        assert repr_module is not None

    def test_cnn_4x1_creation(self):
        """Creates CNN 4x1 representation."""
        repr_module = create_representation("cnn_4x1", {"cnn_channels": 64})
        assert repr_module is not None

    def test_cnn_multi_creation(self):
        """Creates CNN multi representation."""
        repr_module = create_representation("cnn_multi", {"cnn_channels": 64})
        assert repr_module is not None

    def test_invalid_type_raises(self):
        """Invalid representation type raises ValueError."""
        with pytest.raises(ValueError):
            create_representation("invalid", {})


class TestAlgorithmParams:
    """Tests for algorithm parameter definitions."""

    def test_all_algorithms_have_params(self):
        """All algorithms have parameter definitions."""
        expected_algorithms = [
            "dqn", "double_dqn", "dueling_dqn", "per_dqn", "nstep_dqn",
            "rainbow_lite", "reinforce", "a2c", "a3c", "ppo_gae",
            "ppo_value_clip", "acer", "impala", "sarsa", "expected_sarsa",
            "sarsa_lambda", "c51", "qr_dqn", "mcts_learned", "muzero_style"
        ]

        for algo in expected_algorithms:
            assert algo in ALGORITHM_PARAMS, f"Missing params for {algo}"

    def test_all_representations_have_params(self):
        """All representations have parameter definitions."""
        expected_representations = ["onehot", "embedding", "cnn_2x2", "cnn_4x1", "cnn_multi"]

        for repr_type in expected_representations:
            assert repr_type in REPRESENTATION_PARAMS


class TestObjectiveFactory:
    """Tests for ObjectiveFactory class."""

    def test_factory_creation(self):
        """ObjectiveFactory can be created."""
        factory = ObjectiveFactory()
        assert factory is not None

    def test_create_objective_returns_callable(self):
        """create_objective returns a callable."""
        factory = ObjectiveFactory()
        config = SweepStudyConfig(
            study_name="test",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )

        objective = factory.create_objective(config)
        assert callable(objective)

    def test_create_objective_for_different_algorithms(self):
        """Can create objectives for different algorithms."""
        factory = ObjectiveFactory()

        for algo in ["dqn", "double_dqn", "a2c", "ppo_gae", "reinforce"]:
            config = SweepStudyConfig(
                study_name=f"{algo}_test",
                algorithm=algo,
                representation="onehot",
                reward_type="merge",
            )
            objective = factory.create_objective(config)
            assert callable(objective)
