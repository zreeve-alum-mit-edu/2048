"""
Tests for Study Factory.

Tests the generation of all 200 study configurations.

Per DEC-0009: Experimental matrix is (Algorithm, Representation, Reward) tuples.
Per DEC-0011: Each combo gets its own study.
Per DEC-0021: Milestone 26 scope.
"""

import pytest

from sweep.study_factory import (
    StudyFactory,
    SweepStudyConfig,
    ALGORITHMS,
    REPRESENTATIONS,
    REWARD_TYPES,
    get_algorithm_category,
    get_algorithm_family,
)


class TestStudyFactory:
    """Tests for StudyFactory class."""

    def test_total_studies_count(self):
        """Factory generates correct total study count."""
        factory = StudyFactory()
        # 20 algorithms x 5 representations x 2 rewards = 200
        assert factory.total_studies == 200

    def test_generates_all_configs(self):
        """Factory generates all study configurations."""
        factory = StudyFactory()
        configs = factory.generate_all_configs()
        assert len(configs) == 200

    def test_all_configs_are_sweep_study_config(self):
        """All generated configs are SweepStudyConfig instances."""
        factory = StudyFactory()
        configs = factory.generate_all_configs()
        for config in configs:
            assert isinstance(config, SweepStudyConfig)

    def test_unique_study_names(self):
        """All study names are unique."""
        factory = StudyFactory()
        configs = factory.generate_all_configs()
        names = [c.study_name for c in configs]
        assert len(names) == len(set(names))

    def test_study_name_format(self):
        """Study names follow expected format: algo_repr_reward."""
        factory = StudyFactory()
        configs = factory.generate_all_configs()
        for config in configs:
            parts = config.study_name.split("_")
            # Name format should be algorithm_representation_reward
            assert len(parts) >= 3
            assert config.algorithm in config.study_name
            assert config.representation in config.study_name
            assert config.reward_type in config.study_name

    def test_all_algorithms_included(self):
        """All 20 algorithms are included."""
        factory = StudyFactory()
        configs = factory.generate_all_configs()
        algorithms = set(c.algorithm for c in configs)
        assert algorithms == set(ALGORITHMS)
        assert len(algorithms) == 20

    def test_all_representations_included(self):
        """All 5 representations are included."""
        factory = StudyFactory()
        configs = factory.generate_all_configs()
        representations = set(c.representation for c in configs)
        assert representations == set(REPRESENTATIONS)
        assert len(representations) == 5

    def test_both_reward_types_included(self):
        """Both reward types are included."""
        factory = StudyFactory()
        configs = factory.generate_all_configs()
        rewards = set(c.reward_type for c in configs)
        assert rewards == set(REWARD_TYPES)
        assert len(rewards) == 2

    def test_each_algo_has_all_repr_reward_combos(self):
        """Each algorithm has all representation x reward combinations."""
        factory = StudyFactory()
        configs = factory.generate_all_configs()

        for algo in ALGORITHMS:
            algo_configs = [c for c in configs if c.algorithm == algo]
            # 5 representations x 2 rewards = 10 per algorithm
            assert len(algo_configs) == 10

    def test_custom_algorithms_subset(self):
        """Factory can use custom algorithm subset."""
        factory = StudyFactory(algorithms=["dqn", "double_dqn"])
        assert factory.total_studies == 2 * 5 * 2  # 20

    def test_custom_representations_subset(self):
        """Factory can use custom representation subset."""
        factory = StudyFactory(representations=["onehot", "embedding"])
        assert factory.total_studies == 20 * 2 * 2  # 80

    def test_custom_rewards_subset(self):
        """Factory can use single reward type."""
        factory = StudyFactory(reward_types=["merge"])
        assert factory.total_studies == 20 * 5 * 1  # 100

    def test_get_config_by_name(self):
        """Can retrieve config by study name."""
        factory = StudyFactory()
        config = factory.get_config("dqn_onehot_merge")
        assert config is not None
        assert config.algorithm == "dqn"
        assert config.representation == "onehot"
        assert config.reward_type == "merge"

    def test_get_config_invalid_name_returns_none(self):
        """Returns None for invalid study name."""
        factory = StudyFactory()
        config = factory.get_config("invalid_name")
        assert config is None

    def test_get_configs_by_algorithm(self):
        """Can filter configs by algorithm."""
        factory = StudyFactory()
        dqn_configs = factory.get_configs_by_algorithm("dqn")
        assert len(dqn_configs) == 10  # 5 repr x 2 rewards
        assert all(c.algorithm == "dqn" for c in dqn_configs)

    def test_get_configs_by_representation(self):
        """Can filter configs by representation."""
        factory = StudyFactory()
        onehot_configs = factory.get_configs_by_representation("onehot")
        assert len(onehot_configs) == 40  # 20 algo x 2 rewards
        assert all(c.representation == "onehot" for c in onehot_configs)

    def test_summary(self):
        """Summary returns expected information."""
        factory = StudyFactory()
        summary = factory.summary()
        assert summary["total_studies"] == 200
        assert summary["n_trials_per_study"] == 50
        assert len(summary["algorithms"]) == 20
        assert len(summary["representations"]) == 5
        assert len(summary["reward_types"]) == 2


class TestSweepStudyConfig:
    """Tests for SweepStudyConfig dataclass."""

    def test_default_values(self):
        """Config has correct default values."""
        config = SweepStudyConfig(
            study_name="test",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
        )
        assert config.n_trials == 50
        assert config.epochs_per_trial == 300
        assert config.steps_per_epoch == 2500
        assert config.eval_games_per_epoch == 50
        assert config.n_parallel_trials == 1

    def test_custom_values(self):
        """Config accepts custom values."""
        config = SweepStudyConfig(
            study_name="test",
            algorithm="dqn",
            representation="onehot",
            reward_type="merge",
            n_trials=10,
            epochs_per_trial=50,
        )
        assert config.n_trials == 10
        assert config.epochs_per_trial == 50


class TestAlgorithmHelpers:
    """Tests for algorithm helper functions."""

    def test_algorithm_categories(self):
        """Algorithm category function returns correct tier."""
        assert get_algorithm_category("dqn") == "tier_1"
        assert get_algorithm_category("ppo_gae") == "tier_2"
        assert get_algorithm_category("impala") == "tier_3"
        assert get_algorithm_category("muzero_style") == "tier_4"
        assert get_algorithm_category("unknown") == "unknown"

    def test_algorithm_families(self):
        """Algorithm family function returns correct type."""
        assert get_algorithm_family("dqn") == "value_based"
        assert get_algorithm_family("reinforce") == "policy_gradient"
        assert get_algorithm_family("a2c") == "actor_critic"
        assert get_algorithm_family("muzero_style") == "model_based"


class TestAlgorithmConstants:
    """Tests for algorithm constants."""

    def test_algorithms_list_length(self):
        """ALGORITHMS list has 20 entries."""
        assert len(ALGORITHMS) == 20

    def test_representations_list_length(self):
        """REPRESENTATIONS list has 5 entries."""
        assert len(REPRESENTATIONS) == 5

    def test_reward_types_list_length(self):
        """REWARD_TYPES list has 2 entries."""
        assert len(REWARD_TYPES) == 2

    def test_all_algorithms_valid_names(self):
        """All algorithm names are valid identifiers."""
        for algo in ALGORITHMS:
            assert algo.isidentifier() or "_" in algo
