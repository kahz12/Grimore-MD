"""OCR language options are explicit, safe and independent of installed tools."""
import subprocess

import pytest

from grimore.ingest.adapters.base import AdapterOptions
from grimore.ingest.adapters.pdf import PdfAdapter
from grimore.utils import ocr
from grimore.utils.config import Config, load_config


@pytest.mark.parametrize("language", ["eng", "spa", "spa+eng", "chi_sim", "eng+spa"])
def test_language_identifiers_keep_order(language):
    assert ocr.validate_ocr_language(language) == language


@pytest.mark.parametrize("language", ["", " spa", "spa+", "eng++spa", "spa --psm 6",
                                     "../../spa", True, None, 42])
def test_invalid_language_is_rejected_by_adapter(language):
    with pytest.raises(ValueError, match="ocr_language"):
        PdfAdapter(ocr_language=language)


def test_config_language_flows_into_per_call_options(tmp_path):
    path = tmp_path / "grimore.toml"
    path.write_text('[ingest]\nocr = true\nocr_language = "spa+eng"\n')
    config = load_config(str(path))
    assert AdapterOptions.from_config(config).ocr_language == "spa+eng"
    assert Config().ingest.ocr is False
    assert Config().ingest.ocr_language == "eng"


@pytest.mark.parametrize("installed,expected", [
    ("eng\nspa", None), ("eng", "spa"), ("spa", "eng"),
])
def test_probe_matches_whole_language_identifiers(monkeypatch, installed, expected):
    def run(command, **kwargs):
        assert command == ["/tesseract", "--list-langs"]
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, 0, "Header\n" + installed, "")

    monkeypatch.setattr(ocr.subprocess, "run", run)
    failure = ocr.check_ocr_languages("/tesseract", "spa+eng")
    if expected:
        assert failure == "Tesseract language data not installed: " + expected
    else:
        assert failure is None


@pytest.mark.parametrize("kind", ["timeout", "oserror", "exit"])
def test_probe_failure_is_bounded_and_actionable(monkeypatch, kind):
    def run(command, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if kind == "oserror":
            raise OSError("missing binary")
        return subprocess.CompletedProcess(command, 1, "", "tool failed")

    monkeypatch.setattr(ocr.subprocess, "run", run)
    assert "Could not inspect" in ocr.check_ocr_languages("/tesseract", "spa")


def test_adapter_call_options_override_language_without_recreation(tmp_path, monkeypatch):
    from grimore.ingest.adapters import pdf

    source = tmp_path / "scan.pdf"
    source.write_bytes(b"placeholder")
    observed = []
    monkeypatch.setitem(pdf._ENGINE_DISPATCH, "pypdf", lambda _path: ({}, [(1, "")]))

    def recognise(path, page, timeout, language):
        observed.append(language)
        return "Texto recuperado"

    monkeypatch.setattr(pdf, "_ocr_page", recognise)
    adapter = PdfAdapter(ocr=True, ocr_language="spa")
    adapter.extract(source, options=AdapterOptions(vault_root=tmp_path))
    adapter.extract(source, options=AdapterOptions(vault_root=tmp_path, ocr_language="spa+eng"))
    assert observed == ["spa", "spa+eng"]
    with pytest.raises(ValueError, match="ocr_language"):
        adapter.extract(source, options=AdapterOptions(vault_root=tmp_path, ocr_language=""))
