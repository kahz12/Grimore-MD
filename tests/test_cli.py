"""CLI regression tests (B-05 threshold validation, B-09 atomic Oracle export)."""
import pytest
from typer.testing import CliRunner

from grimore.cli import app
from grimore.utils.config import (
    CognitionConfig,
    Config,
    MaintenanceConfig,
    MemoryConfig,
    OutputConfig,
    VaultConfig,
)


@pytest.fixture
def runner():
    return CliRunner()


# ── B-05: --threshold range ────────────────────────────────────────────────


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


# ── B-09: ask --export uses atomic_write ───────────────────────────────────


def _make_export_config(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return Config(
        vault=VaultConfig(path=str(vault_dir), ignored_dirs=[]),
        cognition=CognitionConfig(),
        memory=MemoryConfig(db_path=str(tmp_path / "grimore.db")),
        output=OutputConfig(auto_commit=False, dry_run=True),
        maintenance=MaintenanceConfig(),
    )


class _FakeOracle:
    """Stand-in for the real Oracle so tests don't need Ollama running."""
    def __init__(self, *args, **kwargs):
        pass

    def ask(self, question, top_k=5):
        return {
            "answer": "An exported response from the Oracle.",
            "sources": ["Note Alpha", "Note Beta"],
        }


def _patch_ask_dependencies(monkeypatch, config):
    """Mock everything `ask` builds so only the export path is exercised.

    Services are now instantiated lazily by ``grimore.session.Session``,
    so the patches target that module instead of ``grimore.cli``.
    """
    monkeypatch.setattr("grimore.cli.load_config", lambda: config)
    monkeypatch.setattr("grimore.cli._preflight_or_exit", lambda *a, **k: None)
    monkeypatch.setattr("grimore.session.Database", lambda *a, **k: object())
    monkeypatch.setattr("grimore.session.LLMRouter", lambda *a, **k: object())
    monkeypatch.setattr("grimore.session.Embedder", lambda *a, **k: object())
    monkeypatch.setattr("grimore.session.Oracle", _FakeOracle)


def test_ask_export_writes_full_document(runner, tmp_path, monkeypatch):
    """B-09: the exported note must contain the full Oracle answer + sources."""
    config = _make_export_config(tmp_path)
    _patch_ask_dependencies(monkeypatch, config)

    result = runner.invoke(app, ["ask", "What is X?", "--export", "out.md"])
    assert result.exit_code == 0, result.output

    export_path = tmp_path / "vault" / "out.md"
    assert export_path.exists()
    content = export_path.read_text(encoding="utf-8")
    # Frontmatter + heading + answer + sources, all in one shot.
    assert content.startswith("---\n")
    assert "An exported response from the Oracle." in content
    assert "[[Note Alpha]]" in content
    assert "[[Note Beta]]" in content


def test_ask_export_leaves_no_temp_artifact(runner, tmp_path, monkeypatch):
    """B-09: atomic_write must rename its tempfile away — no .out.md.<rand> left behind."""
    config = _make_export_config(tmp_path)
    _patch_ask_dependencies(monkeypatch, config)

    result = runner.invoke(app, ["ask", "What is X?", "--export", "out.md"])
    assert result.exit_code == 0, result.output

    vault = tmp_path / "vault"
    leftovers = [p.name for p in vault.iterdir() if p.name.startswith(".out.md.")]
    assert leftovers == []
