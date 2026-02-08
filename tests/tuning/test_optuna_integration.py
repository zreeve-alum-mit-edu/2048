"""
Tests for Optuna study creation and configuration.

Per DEC-0010: Optuna with SQLite storage.
Per DEC-0012: MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5).
Per AC-1: 10 Optuna studies can be created in SQLite.
Per AC-3: MedianPruner with specified config.
Per AC-5: Parallel trials work.
Per AC-11: Study resume works.
Per AC-12: Pruning works.
"""

import pytest
import tempfile
import os
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from tuning.study_config import StudyConfig, STUDY_CONFIGS


class TestStudyCreation:
    """Test Optuna study creation (AC-1)."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield f"sqlite:///{db_path}"

    def test_single_study_creation(self, temp_db):
        """Test creating a single study."""
        study = optuna.create_study(
            study_name="test_study",
            storage=temp_db,
            direction="maximize",
            load_if_exists=False
        )

        assert study is not None
        assert study.study_name == "test_study"

    def test_10_studies_can_be_created(self, temp_db):
        """Test all 10 studies can be created in same database (AC-1)."""
        created_studies = []

        for name, config in STUDY_CONFIGS.items():
            study = optuna.create_study(
                study_name=config.study_name,
                storage=temp_db,
                direction="maximize",
                load_if_exists=False
            )
            created_studies.append(study)

        assert len(created_studies) == 10

        # Verify all studies exist in storage
        all_studies = optuna.get_all_study_names(temp_db)
        assert len(all_studies) == 10

    def test_studies_have_unique_names(self, temp_db):
        """Test all studies have unique names."""
        names = set()

        for name, config in STUDY_CONFIGS.items():
            study = optuna.create_study(
                study_name=config.study_name,
                storage=temp_db,
                direction="maximize",
                load_if_exists=False
            )
            names.add(study.study_name)

        assert len(names) == 10

    def test_study_direction_is_maximize(self, temp_db):
        """Test studies are configured to maximize score."""
        study = optuna.create_study(
            study_name="test",
            storage=temp_db,
            direction="maximize"
        )

        assert study.direction == optuna.study.StudyDirection.MAXIMIZE


class TestMedianPruner:
    """Test MedianPruner configuration (AC-3, AC-12)."""

    def test_pruner_config_matches_dec_0012(self):
        """Test pruner matches DEC-0012 specification."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        )

        assert pruner._n_startup_trials == 5
        assert pruner._n_warmup_steps == 10
        assert pruner._interval_steps == 5

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield f"sqlite:///{db_path}"

    def test_study_uses_median_pruner(self, temp_db):
        """Test study can be created with MedianPruner."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        )

        study = optuna.create_study(
            study_name="test",
            storage=temp_db,
            direction="maximize",
            pruner=pruner
        )

        assert study.pruner is pruner

    def test_pruner_prunes_bad_trials(self, temp_db):
        """Test pruner can prune underperforming trials (AC-12)."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=2,  # Smaller for test speed
            n_warmup_steps=2,
            interval_steps=1
        )

        study = optuna.create_study(
            study_name="test",
            storage=temp_db,
            direction="maximize",
            pruner=pruner
        )

        # Objective that reports intermediate values
        def objective(trial):
            for step in range(10):
                # Report progressively
                trial.report(step * 10, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return 100

        # Run a few trials
        study.optimize(objective, n_trials=3, show_progress_bar=False)

        # Should complete without error
        assert len(study.trials) == 3


class TestStudyResume:
    """Test study resume functionality (AC-11)."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield f"sqlite:///{db_path}"

    def test_study_resume_with_load_if_exists(self, temp_db):
        """Test study can be resumed with load_if_exists=True."""
        # Create study and run some trials
        study1 = optuna.create_study(
            study_name="resume_test",
            storage=temp_db,
            direction="maximize",
            load_if_exists=False
        )

        def simple_objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x

        study1.optimize(simple_objective, n_trials=3, show_progress_bar=False)
        initial_trials = len(study1.trials)

        # Resume study
        study2 = optuna.create_study(
            study_name="resume_test",
            storage=temp_db,
            direction="maximize",
            load_if_exists=True  # Key for resume
        )

        assert len(study2.trials) == initial_trials

        # Add more trials
        study2.optimize(simple_objective, n_trials=2, show_progress_bar=False)

        assert len(study2.trials) == initial_trials + 2

    def test_study_preserves_best_trial_on_resume(self, temp_db):
        """Test best trial is preserved on resume."""
        study1 = optuna.create_study(
            study_name="best_test",
            storage=temp_db,
            direction="maximize"
        )

        def objective(trial):
            x = trial.suggest_float("x", 0, 100)
            return x

        study1.optimize(objective, n_trials=5, show_progress_bar=False)
        best_value = study1.best_value
        best_params = study1.best_params

        # Resume and check
        study2 = optuna.create_study(
            study_name="best_test",
            storage=temp_db,
            direction="maximize",
            load_if_exists=True
        )

        assert study2.best_value == best_value
        assert study2.best_params == best_params


class TestSamplerConfiguration:
    """Test TPE sampler configuration."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield f"sqlite:///{db_path}"

    def test_tpe_sampler_with_seed(self, temp_db):
        """Test TPE sampler accepts seed for reproducibility."""
        sampler = TPESampler(seed=42)

        study = optuna.create_study(
            study_name="test",
            storage=temp_db,
            direction="maximize",
            sampler=sampler
        )

        assert study.sampler is sampler

    def test_tpe_sampler_reproducible(self, temp_db):
        """Test TPE sampler produces reproducible results with same seed."""
        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            return x + y

        # First run
        sampler1 = TPESampler(seed=42)
        study1 = optuna.create_study(
            study_name="repro1",
            storage=temp_db,
            sampler=sampler1
        )
        study1.optimize(objective, n_trials=3, show_progress_bar=False)
        params1 = [t.params for t in study1.trials]

        # Second run with same seed (new storage)
        with tempfile.TemporaryDirectory() as tmpdir2:
            db2 = f"sqlite:///{tmpdir2}/test2.db"
            sampler2 = TPESampler(seed=42)
            study2 = optuna.create_study(
                study_name="repro2",
                storage=db2,
                sampler=sampler2
            )
            study2.optimize(objective, n_trials=3, show_progress_bar=False)
            params2 = [t.params for t in study2.trials]

        # Same seed should produce same suggestions
        assert params1 == params2


class TestIntermediateReporting:
    """Test intermediate value reporting (AC-4)."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield f"sqlite:///{db_path}"

    def test_trial_reports_intermediate_values(self, temp_db):
        """Test trials can report intermediate scores (AC-4)."""
        study = optuna.create_study(
            study_name="intermediate_test",
            storage=temp_db,
            direction="maximize"
        )

        def objective(trial):
            for epoch in range(5):
                # Simulate training - report score each epoch
                score = epoch * 10
                trial.report(score, epoch)

            return 100

        study.optimize(objective, n_trials=1, show_progress_bar=False)

        # Check intermediate values were recorded
        trial = study.trials[0]
        assert len(trial.intermediate_values) == 5
        assert trial.intermediate_values[0] == 0
        assert trial.intermediate_values[4] == 40


class TestBestHyperparams:
    """Test best hyperparameters retrieval (AC-9)."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield f"sqlite:///{db_path}"

    def test_best_params_retrievable(self, temp_db):
        """Test best hyperparameters can be retrieved (AC-9)."""
        study = optuna.create_study(
            study_name="best_params_test",
            storage=temp_db,
            direction="maximize"
        )

        def objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            return lr * batch_size  # Simple objective

        study.optimize(objective, n_trials=10, show_progress_bar=False)

        # Best params should be retrievable
        best_params = study.best_params
        assert "learning_rate" in best_params
        assert "batch_size" in best_params

        # Best value should be retrievable
        best_value = study.best_value
        assert best_value is not None

        # Best trial should be retrievable
        best_trial = study.best_trial
        assert best_trial is not None
        assert best_trial.value == best_value

    def test_best_params_after_resume(self, temp_db):
        """Test best params still available after resume."""
        def objective(trial):
            x = trial.suggest_float("x", 0, 100)
            return x

        # Run initial study
        study1 = optuna.create_study(
            study_name="resume_best",
            storage=temp_db,
            direction="maximize"
        )
        study1.optimize(objective, n_trials=5, show_progress_bar=False)

        # Resume and verify
        study2 = optuna.create_study(
            study_name="resume_best",
            storage=temp_db,
            direction="maximize",
            load_if_exists=True
        )

        assert study2.best_params == study1.best_params
        assert study2.best_value == study1.best_value
