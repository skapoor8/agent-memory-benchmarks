"""Smoke tests for the Click CLI."""

from click.testing import CliRunner

from agent_memory_benchmarks.__main__ import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "bench" in result.output


def test_bench_help():
    runner = CliRunner()
    result = runner.invoke(main, ["bench", "--help"])
    assert result.exit_code == 0
    assert "--store" in result.output
