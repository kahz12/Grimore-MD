"""Unextractable documents remain in the fixed evaluation denominator."""
import copy
import shutil

import pytest
import yaml

from eval.check_extraction import DEFAULT_MANIFEST, EVALUATION_SEMANTICS, evaluate, load_manifest
from eval.benchmark_extraction import benchmark


def _manifest_with_failure(tmp_path, *, include_healthy):
    root, documents = load_manifest(DEFAULT_MANIFEST)
    healthy = copy.deepcopy(documents[0])
    shutil.copyfile(root / healthy["path"], tmp_path / healthy["path"])
    broken = copy.deepcopy(healthy)
    broken["id"], broken["path"] = "unreadable-document", "unreadable.pdf"
    (tmp_path / broken["path"]).write_bytes(b"This is not a PDF.\n")
    items = [healthy, broken] if include_healthy else [broken]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({
        "version": 1, "corpus_root": str(tmp_path), "documents": items,
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize("include_healthy", [True, False])
def test_unreadable_document_keeps_every_expected_check(tmp_path, include_healthy):
    manifest = _manifest_with_failure(tmp_path, include_healthy=include_healthy)
    report = evaluate(manifest)
    assert report["evaluation_semantics"] == EVALUATION_SEMANTICS
    assert report["passed"] is False
    assert report["totals"] == {
        "reading_order": {"passed": 3 if include_healthy else 0,
                          "total": 6 if include_healthy else 3},
        "table_rows": {"passed": 4 if include_healthy else 0,
                       "total": 8 if include_healthy else 4},
        "page_anchors": {"passed": 6 if include_healthy else 0,
                         "total": 12 if include_healthy else 6},
    }
    failed = report["documents"][-1]
    assert failed["id"] == "unreadable-document"
    assert failed["error"]
    assert len(failed["checks"]) == 13
    assert all(check["passed"] is False for check in failed["checks"])


@pytest.mark.parametrize("broken_first", [True, False])
def test_failed_document_does_not_reuse_previous_pages(tmp_path, broken_first):
    manifest = _manifest_with_failure(tmp_path, include_healthy=True)
    if broken_first:
        data = yaml.safe_load(manifest.read_text())
        data["documents"].reverse()
        manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    report = evaluate(manifest)
    by_id = {document["id"]: document for document in report["documents"]}
    broken = by_id.pop("unreadable-document")
    healthy = next(iter(by_id.values()))
    assert all(check["passed"] for check in healthy["checks"])
    assert not any(check["passed"] for check in broken["checks"])
    assert [(check["kind"], check["index"], check["page"])
            for check in healthy["checks"]] == [
        (check["kind"], check["index"], check["page"])
        for check in broken["checks"]
    ]


def test_worker_reports_execution_success_separately_from_extraction_failure(tmp_path):
    manifest = _manifest_with_failure(tmp_path, include_healthy=True)
    report = benchmark(manifest, engines=["pypdf"], repeat=1)
    measured = report["results"][0]
    assert measured["status"] == "ok"
    assert measured["quality"]["passed"] is False
    assert measured["quality"]["evaluation_semantics"] == EVALUATION_SEMANTICS
    assert measured["quality"]["document_errors"] == ["unreadable-document"]
    assert measured["quality"]["totals"]["page_anchors"] == {"passed": 6, "total": 12}
    assert len(measured["quality"]["failed_checks"]) == 13
    assert all(check["document_id"] == "unreadable-document"
               for check in measured["quality"]["failed_checks"])
    assert report["inputs_unchanged"] is True


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_read_errors_keep_checks_even_if_the_error_message_is_empty(tmp_path, monkeypatch, failure):
    from grimore.ingest.adapters.pdf import PdfAdapter

    manifest = _manifest_with_failure(tmp_path, include_healthy=False)

    def fail(*_args, **_kwargs):
        raise failure()

    monkeypatch.setattr(PdfAdapter, "extract", fail)
    report = evaluate(manifest)
    assert report["passed"] is False
    assert "error" in report["documents"][0]
    assert len(report["documents"][0]["checks"]) == 13
    assert sum(item["total"] for item in report["totals"].values()) == 13
    assert sum(item["passed"] for item in report["totals"].values()) == 0
