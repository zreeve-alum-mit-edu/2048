"""Tests for hyperparameter search spaces."""

import pytest
import optuna

from tuning.search_spaces import suggest_hyperparams, get_default_params


class TestSuggestHyperparams:
    """Test hyperparameter suggestion function."""

    @pytest.fixture
    def mock_trial(self):
        """Create a mock Optuna trial for testing."""
        study = optuna.create_study()
        trial = study.ask()
        return trial

    def test_returns_dict(self, mock_trial):
        """Test function returns a dictionary."""
        params = suggest_hyperparams(mock_trial, "onehot")
        assert isinstance(params, dict)

    def test_training_params_present(self, mock_trial):
        """Test training hyperparameters are included."""
        params = suggest_hyperparams(mock_trial, "onehot")
        assert "learning_rate" in params
        assert "batch_size" in params
        assert "gamma" in params

    def test_epsilon_params_present(self, mock_trial):
        """Test epsilon schedule parameters are included (DEC-0035)."""
        params = suggest_hyperparams(mock_trial, "onehot")
        assert params["epsilon_start"] == 1.0  # Fixed per DEC-0035
        assert "epsilon_end" in params
        assert "epsilon_decay_steps" in params

    def test_target_update_present(self, mock_trial):
        """Test target network param is included (DEC-0036)."""
        params = suggest_hyperparams(mock_trial, "onehot")
        assert "target_update_frequency" in params

    def test_buffer_params_present(self, mock_trial):
        """Test replay buffer params are included."""
        params = suggest_hyperparams(mock_trial, "onehot")
        assert "buffer_capacity" in params
        assert "buffer_min_size" in params

    def test_hidden_layers_present(self, mock_trial):
        """Test network architecture params are included."""
        params = suggest_hyperparams(mock_trial, "onehot")
        assert "hidden_layers" in params
        assert isinstance(params["hidden_layers"], list)
        assert len(params["hidden_layers"]) >= 1

    def test_embed_dim_for_embedding(self, mock_trial):
        """Test embed_dim is suggested for embedding representation."""
        params = suggest_hyperparams(mock_trial, "embedding")
        assert "embed_dim" in params

    def test_cnn_channels_for_cnn(self, mock_trial):
        """Test cnn_channels is suggested for CNN representations."""
        for repr_type in ["cnn_2x2", "cnn_4x1", "cnn_multi"]:
            study = optuna.create_study()
            trial = study.ask()
            params = suggest_hyperparams(trial, repr_type)
            assert "cnn_channels" in params

    def test_no_embed_dim_for_onehot(self, mock_trial):
        """Test embed_dim is not suggested for onehot."""
        params = suggest_hyperparams(mock_trial, "onehot")
        assert "embed_dim" not in params

    def test_learning_rate_in_range(self, mock_trial):
        """Test learning rate is in expected range."""
        params = suggest_hyperparams(mock_trial, "onehot")
        assert 1e-5 <= params["learning_rate"] <= 1e-2


class TestGetDefaultParams:
    """Test default parameter getter."""

    def test_returns_dict(self):
        """Test function returns a dictionary."""
        params = get_default_params("onehot")
        assert isinstance(params, dict)

    def test_onehot_defaults(self):
        """Test default params for onehot."""
        params = get_default_params("onehot")
        assert params["learning_rate"] == 0.0001
        assert params["batch_size"] == 64
        assert params["gamma"] == 0.99
        assert "embed_dim" not in params

    def test_embedding_defaults(self):
        """Test default params for embedding."""
        params = get_default_params("embedding")
        assert "embed_dim" in params

    def test_cnn_defaults(self):
        """Test default params for CNN."""
        for repr_type in ["cnn_2x2", "cnn_4x1", "cnn_multi"]:
            params = get_default_params(repr_type)
            assert "cnn_channels" in params
