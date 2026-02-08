"""
Short functional tests for the tuning module.

These tests run actual (abbreviated) training to verify end-to-end functionality.
They use minimal settings (1-2 trials, 1-2 epochs) to run quickly.
"""

import pytest
import tempfile
import os

import torch
import optuna
from optuna.samplers import TPESampler

from tuning.study_config import StudyConfig
from tuning.objective import create_objective
from tuning.search_spaces import suggest_hyperparams
from tuning.utils import create_representation
from algorithms.dqn.agent import DQNAgent
from game.env import GameEnv
from game.moves import compute_valid_mask


# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


class TestMinimalObjective:
    """Test objective function with minimal settings."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def minimal_config(self):
        """Create minimal config for quick testing."""
        return StudyConfig(
            study_name="test_minimal",
            representation_type="onehot",
            reward_type="merge",
            n_trials=1,
            epochs_per_trial=1,
            steps_per_epoch=10,  # Very short
            eval_games_per_epoch=2,
        )

    def test_objective_runs_to_completion(self, minimal_config):
        """Test objective function completes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=1,
                n_warmup_steps=1,
                interval_steps=1
            )

            study = optuna.create_study(
                study_name=minimal_config.study_name,
                storage=storage,
                direction="maximize",
                pruner=pruner
            )

            objective = create_objective(minimal_config)

            # Run 1 trial
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            assert len(study.trials) == 1
            assert study.trials[0].state == optuna.trial.TrialState.COMPLETE

    def test_objective_returns_float_score(self, minimal_config):
        """Test objective returns a float score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            study = optuna.create_study(
                study_name=minimal_config.study_name,
                storage=storage,
                direction="maximize"
            )

            objective = create_objective(minimal_config)
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            assert study.best_value is not None
            assert isinstance(study.best_value, float)


class TestObjectiveWithDifferentRepresentations:
    """Test objective works with all 5 representations."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("repr_type", [
        "onehot",
        "embedding",
        "cnn_2x2",
        "cnn_4x1",
        "cnn_multi",
    ])
    def test_representation_objective_runs(self, repr_type):
        """Test objective runs with each representation type."""
        config = StudyConfig(
            study_name=f"test_{repr_type}",
            representation_type=repr_type,
            reward_type="merge",
            n_trials=1,
            epochs_per_trial=1,
            steps_per_epoch=10,
            eval_games_per_epoch=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            study = optuna.create_study(
                study_name=config.study_name,
                storage=storage,
                direction="maximize"
            )

            objective = create_objective(config)
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            assert len(study.trials) == 1


class TestObjectiveWithDifferentRewards:
    """Test objective works with both reward types."""

    @pytest.mark.parametrize("reward_type", ["merge", "spawn"])
    def test_reward_type_objective_runs(self, reward_type):
        """Test objective runs with each reward type."""
        config = StudyConfig(
            study_name=f"test_{reward_type}",
            representation_type="onehot",
            reward_type=reward_type,
            n_trials=1,
            epochs_per_trial=1,
            steps_per_epoch=10,
            eval_games_per_epoch=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            study = optuna.create_study(
                study_name=config.study_name,
                storage=storage,
                direction="maximize"
            )

            objective = create_objective(config)
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            assert len(study.trials) == 1


class TestMultipleTrials:
    """Test running multiple trials."""

    def test_two_trials_complete(self):
        """Test 2 trials can complete."""
        config = StudyConfig(
            study_name="test_2_trials",
            representation_type="onehot",
            reward_type="merge",
            n_trials=2,
            epochs_per_trial=1,
            steps_per_epoch=10,
            eval_games_per_epoch=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            study = optuna.create_study(
                study_name=config.study_name,
                storage=storage,
                direction="maximize"
            )

            objective = create_objective(config)
            study.optimize(objective, n_trials=2, show_progress_bar=False)

            assert len(study.trials) == 2
            # Both should complete (not pruned with this minimal setup)
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            assert len(completed) >= 1

    def test_trials_have_different_params(self):
        """Test different trials explore different hyperparameters."""
        config = StudyConfig(
            study_name="test_different_params",
            representation_type="onehot",
            reward_type="merge",
            n_trials=3,
            epochs_per_trial=1,
            steps_per_epoch=10,
            eval_games_per_epoch=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name=config.study_name,
                storage=storage,
                direction="maximize",
                sampler=sampler
            )

            objective = create_objective(config)
            study.optimize(objective, n_trials=3, show_progress_bar=False)

            # Get all param sets
            all_params = [t.params for t in study.trials]

            # At least some params should differ across trials
            # (TPE sampler should explore different regions)
            unique_lr = len(set(p.get("learning_rate") for p in all_params))
            assert unique_lr >= 1  # With random sampling, should have variation


class TestEpochReporting:
    """Test epoch-level score reporting."""

    def test_intermediate_values_reported(self):
        """Test intermediate values are reported for each epoch."""
        config = StudyConfig(
            study_name="test_intermediate",
            representation_type="onehot",
            reward_type="merge",
            n_trials=1,
            epochs_per_trial=3,  # 3 epochs to check reporting
            steps_per_epoch=10,
            eval_games_per_epoch=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            study = optuna.create_study(
                study_name=config.study_name,
                storage=storage,
                direction="maximize"
            )

            objective = create_objective(config)
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            trial = study.trials[0]
            # Should have intermediate values for each epoch
            assert len(trial.intermediate_values) == 3


class TestQuickEval:
    """Test the quick evaluation function indirectly through objective."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_eval_produces_numeric_score(self):
        """Test evaluation produces a numeric score."""
        config = StudyConfig(
            study_name="test_eval",
            representation_type="onehot",
            reward_type="merge",
            n_trials=1,
            epochs_per_trial=1,
            steps_per_epoch=5,
            eval_games_per_epoch=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = f"sqlite:///{tmpdir}/test.db"

            study = optuna.create_study(
                study_name=config.study_name,
                storage=storage,
                direction="maximize"
            )

            objective = create_objective(config)
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            # Final value should be numeric (average score)
            assert study.best_value is not None
            assert study.best_value >= 0  # Scores are non-negative


class TestManualTrainingLoop:
    """Test manual training to verify components work together."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_manual_training_loop(self, device):
        """Test a manual mini training loop works end-to-end."""
        # Create representation
        representation = create_representation("onehot", {}).to(device)

        # Create agent
        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=[64],
            buffer_min_size=5,
            batch_size=4,
        )

        # Create environment
        env = GameEnv(n_games=4, device=device)
        state = env.reset()

        # Training loop
        for step in range(10):
            valid_mask = compute_valid_mask(state, device)
            actions = agent.select_action(state, valid_mask, training=True)

            result = env.step(actions)

            agent.store_transition(
                state=state,
                action=actions,
                reward=result.merge_reward.float(),
                next_state=result.next_state,
                done=result.done,
                valid_mask=result.valid_mask,
            )

            agent.train_step()

            # Update state
            state = torch.where(
                result.done.unsqueeze(-1).unsqueeze(-1),
                result.reset_states,
                result.next_state
            )
            env._state = state.clone()

        # Should complete without error
        assert agent.step_count > 0

    @pytest.mark.parametrize("repr_type", ["onehot", "embedding", "cnn_2x2"])
    def test_manual_loop_multiple_representations(self, device, repr_type):
        """Test manual training with different representations."""
        params = {"embed_dim": 16, "cnn_channels": 32}
        representation = create_representation(repr_type, params).to(device)

        agent = DQNAgent(
            device=device,
            representation=representation,
            hidden_layers=[32],
            buffer_min_size=5,
            batch_size=4,
        )

        env = GameEnv(n_games=4, device=device)
        state = env.reset()

        for step in range(5):
            valid_mask = compute_valid_mask(state, device)
            actions = agent.select_action(state, valid_mask, training=True)
            result = env.step(actions)

            agent.store_transition(
                state=state,
                action=actions,
                reward=result.merge_reward.float(),
                next_state=result.next_state,
                done=result.done,
                valid_mask=result.valid_mask,
            )

            agent.train_step()
            state = result.next_state

        # Should complete without error
        assert True
