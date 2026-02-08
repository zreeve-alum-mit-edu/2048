"""Integration tests for tuning module."""

import pytest
import torch
import optuna
from pathlib import Path
import tempfile

from tuning.study_config import StudyConfig
from tuning.utils import create_representation
from tuning.search_spaces import get_default_params
from algorithms.dqn.agent import DQNAgent


class TestDQNWithRepresentations:
    """Test DQN works with all representations (AC-7)."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_dqn_with_onehot(self, device):
        """Test DQN agent with OneHot representation."""
        repr_module = create_representation("onehot", {})
        agent = DQNAgent(device=device, representation=repr_module)
        assert agent.policy_net.input_size == 272

    def test_dqn_with_embedding(self, device):
        """Test DQN agent with Embedding representation."""
        repr_module = create_representation("embedding", {"embed_dim": 32})
        agent = DQNAgent(device=device, representation=repr_module)
        assert agent.policy_net.input_size == 512

    def test_dqn_with_cnn_2x2(self, device):
        """Test DQN agent with CNN-2x2 representation."""
        repr_module = create_representation("cnn_2x2", {"cnn_channels": 64})
        agent = DQNAgent(device=device, representation=repr_module)
        assert agent.policy_net.input_size == 576

    def test_dqn_with_cnn_4x1(self, device):
        """Test DQN agent with CNN-4x1 representation."""
        repr_module = create_representation("cnn_4x1", {"cnn_channels": 64})
        agent = DQNAgent(device=device, representation=repr_module)
        # 4x1 on 4x4 -> 1x4, 1x4 on 4x4 -> 4x1, concat = 64*4 + 64*4 = 512
        assert agent.policy_net.input_size == 512

    def test_dqn_with_cnn_multi(self, device):
        """Test DQN agent with CNN-Multi representation."""
        repr_module = create_representation("cnn_multi", {"cnn_channels": 64})
        agent = DQNAgent(device=device, representation=repr_module)
        # 2x2->576, 4x1->256, 1x4->256, total = 1088
        expected = 64 * 9 + 64 * 4 + 64 * 4  # 576 + 256 + 256 = 1088
        assert agent.policy_net.input_size == expected

    def test_dqn_forward_all_representations(self, device):
        """Test DQN can do forward pass with all representations."""
        batch = torch.zeros(4, 16, 17, dtype=torch.bool, device=device)
        batch[:, :, 0] = True
        valid_mask = torch.ones(4, 4, dtype=torch.bool, device=device)

        params_map = {
            "onehot": {},
            "embedding": {"embed_dim": 32},
            "cnn_2x2": {"cnn_channels": 64},
            "cnn_4x1": {"cnn_channels": 64},
            "cnn_multi": {"cnn_channels": 64},
        }

        for repr_type, params in params_map.items():
            repr_module = create_representation(repr_type, params)
            agent = DQNAgent(device=device, representation=repr_module)
            actions = agent.select_action(batch, valid_mask, training=False)
            assert actions.shape == (4,)


class TestOptunaStudyCreation:
    """Test Optuna study creation (AC-1, AC-3)."""

    def test_study_creation_with_pruner(self):
        """Test study can be created with MedianPruner (DEC-0012)."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"
            study = optuna.create_study(
                study_name="test_study",
                storage=storage,
                direction="maximize",
                pruner=pruner
            )
            assert study.study_name == "test_study"

    def test_study_resume(self):
        """Test study can be resumed from SQLite (AC-11)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            # Create study
            study1 = optuna.create_study(
                study_name="resume_test",
                storage=storage,
                direction="maximize"
            )
            study1.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)

            # Resume study
            study2 = optuna.create_study(
                study_name="resume_test",
                storage=storage,
                direction="maximize",
                load_if_exists=True
            )
            assert len(study2.trials) == 2


class TestPrunerConfiguration:
    """Test pruner configuration matches DEC-0012."""

    def test_median_pruner_config(self):
        """Test MedianPruner has correct config."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        )
        # Verify pruner is created without error
        assert isinstance(pruner, optuna.pruners.MedianPruner)
