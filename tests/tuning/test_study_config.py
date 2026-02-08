"""Tests for study configuration."""

import pytest

from tuning.study_config import StudyConfig, STUDY_CONFIGS


class TestStudyConfig:
    """Test StudyConfig dataclass."""

    def test_study_config_defaults(self):
        """Test default values are correct per DEC-0037."""
        config = StudyConfig(
            study_name="test_study",
            representation_type="onehot",
            reward_type="merge"
        )
        assert config.n_trials == 50
        assert config.epochs_per_trial == 300
        assert config.steps_per_epoch == 2500
        assert config.eval_games_per_epoch == 50
        assert config.n_parallel_trials == 4
        assert "sqlite" in config.storage_path

    def test_study_config_custom_values(self):
        """Test custom values override defaults."""
        config = StudyConfig(
            study_name="test",
            representation_type="embedding",
            reward_type="spawn",
            n_trials=10,
            epochs_per_trial=5
        )
        assert config.n_trials == 10
        assert config.epochs_per_trial == 5


class TestSTUDY_CONFIGS:
    """Test predefined study configurations."""

    def test_study_configs_count(self):
        """Test 10 studies exist (5 repr x 2 rewards)."""
        assert len(STUDY_CONFIGS) == 10

    def test_all_representations_present(self):
        """Test all 5 representation types are covered."""
        repr_types = {c.representation_type for c in STUDY_CONFIGS.values()}
        expected = {"onehot", "embedding", "cnn_2x2", "cnn_4x1", "cnn_multi"}
        assert repr_types == expected

    def test_both_reward_types_present(self):
        """Test both reward types are covered."""
        reward_types = {c.reward_type for c in STUDY_CONFIGS.values()}
        assert reward_types == {"merge", "spawn"}

    def test_each_repr_has_both_rewards(self):
        """Test each representation has both merge and spawn variants."""
        # Check by representation_type in the configs
        for repr_type in ["onehot", "embedding", "cnn_2x2", "cnn_4x1", "cnn_multi"]:
            merge_found = any(
                c.representation_type == repr_type and c.reward_type == "merge"
                for c in STUDY_CONFIGS.values()
            )
            spawn_found = any(
                c.representation_type == repr_type and c.reward_type == "spawn"
                for c in STUDY_CONFIGS.values()
            )
            assert merge_found, f"Missing merge variant for {repr_type}"
            assert spawn_found, f"Missing spawn variant for {repr_type}"

    def test_study_names_unique(self):
        """Test all study names are unique."""
        names = [c.study_name for c in STUDY_CONFIGS.values()]
        assert len(names) == len(set(names))
