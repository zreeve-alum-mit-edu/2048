"""Integration tests for REINFORCE algorithm."""

import pytest
import torch

from algorithms.reinforce.run import train, evaluate
from algorithms.reinforce.agent import REINFORCEAgent
from game.env import GameEnv


class TestTrainFunction:
    """Test the train function interface (DEC-0006)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def env_factory(self, device):
        def factory():
            return GameEnv(n_games=4, device=device)
        return factory

    def test_train_runs_without_error(self, env_factory, tmp_path):
        """Test training runs without errors (basic smoke test)."""
        # Create a minimal config for fast testing
        config_content = """
training:
  total_steps: 100
  learning_rate: 0.0001
  gamma: 0.99
  update_frequency: 4
network:
  hidden_layers: [32, 32]
  activation: relu
env:
  n_games: 4
checkpoint:
  save_frequency: 50
  save_dir: {checkpoint_dir}
logging:
  log_frequency: 50
  eval_frequency: 50
  eval_games: 4
""".format(checkpoint_dir=str(tmp_path / "checkpoints"))

        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        result = train(env_factory, str(config_path))

        assert result is not None
        assert len(result.checkpoints) > 0
        assert "total_steps" in result.metrics


class TestEvaluateFunction:
    """Test the evaluate function interface (DEC-0006)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def env_factory(self, device):
        def factory():
            return GameEnv(n_games=4, device=device)
        return factory

    def test_evaluate_with_agent(self, env_factory, device):
        """Test evaluation with pre-loaded agent."""
        agent = REINFORCEAgent(device=device, hidden_layers=[32, 32])

        result = evaluate(env_factory, None, num_games=4, agent=agent)

        assert result is not None
        assert len(result.scores) == 4
        assert result.avg_score >= 0
        assert result.max_score >= 0

    def test_evaluate_returns_correct_count(self, env_factory, device):
        """Test evaluation returns exactly num_games scores."""
        agent = REINFORCEAgent(device=device, hidden_layers=[32, 32])

        for num_games in [4, 8, 16]:
            result = evaluate(env_factory, None, num_games=num_games, agent=agent)
            assert len(result.scores) == num_games


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_agent_collects_and_trains(self, device):
        """Test that agent can collect episodes and train."""
        agent = REINFORCEAgent(
            device=device,
            hidden_layers=[32, 32],
        )

        env = GameEnv(n_games=8, device=device)
        state = env.reset()

        # Run until we complete some episodes
        episodes_completed = 0
        max_steps = 200

        for step in range(max_steps):
            from game.moves import compute_valid_mask
            valid_mask = compute_valid_mask(state, device)
            actions = agent.select_action(state, valid_mask, training=True)
            result = env.step(actions)

            num_complete = agent.store_transition(
                state=state,
                action=actions,
                reward=result.merge_reward.float(),
                done=result.done,
            )

            if num_complete > episodes_completed:
                episodes_completed = num_complete

            state = torch.where(
                result.done.unsqueeze(-1).unsqueeze(-1),
                result.reset_states,
                result.next_state
            )
            env._state = state.clone()

            if episodes_completed >= 4:
                break

        # Should have completed some episodes
        assert episodes_completed > 0

        # Train and verify we get metrics
        metrics = agent.train_step(min_trajectories=1)
        if metrics is not None:
            assert "loss" in metrics
