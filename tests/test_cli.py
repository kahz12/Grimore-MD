"""B-05 regression: --threshold for `grimoire connect` is bounded to [0.0, 1.0]."""
import pytest
from typer.testing import CliRunner

from grimoire.cli import app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize("bad", ["-10", "-0.001", "1.001", "5", "100"])
def test_threshold_out_of_range_is_rejected(runner, bad):
    result = runner.invoke(app, ["connect", "--threshold", bad])
    assert result.exit_code != 0
    assert "must be in [0.0, 1.0]" in (result.output + (result.stderr or ""))


@pytest.mark.parametrize("good", ["0.0", "0.5", "1.0"])
def test_threshold_in_range_passes_validation(runner, good):
    """In-range values must not trigger the BadParameter from _validate_threshold.

    The command body may still fail later (e.g. missing DB in a clean tmpdir),
    so we only assert the validator-level error is absent.
    """
    result = runner.invoke(app, ["connect", "--threshold", good])
    assert "must be in [0.0, 1.0]" not in (result.output + (result.stderr or ""))
