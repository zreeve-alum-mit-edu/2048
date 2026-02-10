"""Tests for orchestrator CLI module.

Tests the command-line interface entry point.
"""

import pytest
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from orchestrator.__main__ import main, cmd_list, cmd_quick
from orchestrator.config import save_config, ExperimentConfig, OrchestratorConfig


class TestCLIMain:
    """Tests for CLI main entry point."""

    def test_no_command_shows_help(self, capsys):
        """Test that no command shows help and exits."""
        with patch.object(sys, 'argv', ['orchestrator']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_help_flag(self, capsys):
        """Test help flag shows usage."""
        with patch.object(sys, 'argv', ['orchestrator', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Training Orchestrator" in captured.out
        assert "run" in captured.out
        assert "report" in captured.out


class TestCLIList:
    """Tests for CLI list command."""

    def test_list_config(self, capsys):
        """Test listing experiments in a config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Create test config
            exp = ExperimentConfig(
                name="test_exp",
                algorithm="dqn",
                representation="onehot",
                reward_type="merge",
                training_steps=50000,
                eval_games=50,
            )
            config = OrchestratorConfig(experiments=[exp])
            save_config(config, str(config_path))

            # Run list command
            with patch.object(sys, 'argv', ['orchestrator', 'list', str(config_path)]):
                main()

            captured = capsys.readouterr()
            assert "test_exp" in captured.out
            assert "dqn" in captured.out
            assert "onehot" in captured.out

    def test_list_nonexistent_config(self, capsys):
        """Test listing non-existent config exits with error."""
        with patch.object(sys, 'argv', ['orchestrator', 'list', '/nonexistent/config.yaml']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "not found" in captured.out


class TestCLIQuick:
    """Tests for CLI quick config generation command."""

    def test_quick_generates_config(self, capsys):
        """Test quick command generates config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quick.yaml"

            with patch.object(sys, 'argv', [
                'orchestrator', 'quick',
                '--algorithms', 'dqn',
                '--representations', 'onehot',
                '--rewards', 'merge',
                '--output', str(output_path),
            ]):
                main()

            assert output_path.exists()

            captured = capsys.readouterr()
            assert "Generated config" in captured.out
            assert "dqn_onehot_merge" in captured.out

    def test_quick_multiple_combinations(self, capsys):
        """Test quick command with multiple combinations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quick.yaml"

            with patch.object(sys, 'argv', [
                'orchestrator', 'quick',
                '--algorithms', 'dqn,reinforce',
                '--representations', 'onehot,embedding',
                '--rewards', 'merge',
                '--output', str(output_path),
            ]):
                main()

            captured = capsys.readouterr()
            assert "4 experiments" in captured.out

    def test_quick_custom_params(self, capsys):
        """Test quick command with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quick.yaml"

            with patch.object(sys, 'argv', [
                'orchestrator', 'quick',
                '--algorithms', 'dqn',
                '--representations', 'onehot',
                '--steps', '25000',
                '--eval-games', '50',
                '--results-dir', 'custom_results',
                '--output', str(output_path),
            ]):
                main()

            assert output_path.exists()

            # Verify config contents
            from orchestrator.config import load_config
            config = load_config(str(output_path))
            assert config.experiments[0].training_steps == 25000
            assert config.experiments[0].eval_games == 50
            assert config.results_dir == "custom_results"


class TestCLIRun:
    """Tests for CLI run command."""

    def test_run_nonexistent_config(self, capsys):
        """Test running with non-existent config exits with error."""
        with patch.object(sys, 'argv', ['orchestrator', 'run', '/nonexistent/config.yaml']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_run_invalid_config(self, capsys):
        """Test running with invalid config exits with error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            config_path.write_text("invalid: yaml: content:")

            with patch.object(sys, 'argv', ['orchestrator', 'run', str(config_path)]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


class TestCLIReport:
    """Tests for CLI report command."""

    def test_report_nonexistent_dir(self, capsys):
        """Test report with non-existent directory exits with error."""
        with patch.object(sys, 'argv', ['orchestrator', 'report', '/nonexistent/results']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_report_empty_dir(self, capsys):
        """Test report with empty results directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sys, 'argv', ['orchestrator', 'report', tmpdir]):
                main()  # Should not raise

            captured = capsys.readouterr()
            assert "Reports generated" in captured.out

    def test_report_format_json(self, capsys):
        """Test report with JSON format only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sys, 'argv', [
                'orchestrator', 'report', tmpdir,
                '--format', 'json',
            ]):
                main()

            # Check only JSON was created
            assert (Path(tmpdir) / "report.json").exists()
            assert not (Path(tmpdir) / "report.md").exists()

    def test_report_format_markdown(self, capsys):
        """Test report with markdown format only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sys, 'argv', [
                'orchestrator', 'report', tmpdir,
                '--format', 'markdown',
            ]):
                main()

            # Check only markdown was created
            assert (Path(tmpdir) / "report.md").exists()
            assert not (Path(tmpdir) / "report.json").exists()

    def test_report_custom_output_dir(self, capsys):
        """Test report with custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()
            output_dir = Path(tmpdir) / "reports"

            with patch.object(sys, 'argv', [
                'orchestrator', 'report', str(results_dir),
                '--output', str(output_dir),
            ]):
                main()

            assert output_dir.exists()
            assert (output_dir / "report.md").exists()
            assert (output_dir / "report.json").exists()
