"""Small local-model control, never a representative or held-out evaluation.

Copies one PDF into an owned temporary vault/index. No scan, tagging, sidecar,
downloads, writes to the user's index, external judge or remote endpoint.
Prints JSON on stdout; progress/diagnostics go to stderr.
"""
# ruff: noqa: E402 -- direct script execution needs the repository on sys.path
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.benchmark_extraction import hardware_profile, _peak_rss_mb
from eval.context_order_control import STRATEGIES, presentation_control
from grimore.cognition.chunker import build_candidate_chunks
from grimore.cognition.eval import load_golden, run_eval, run_provenance
from grimore.cognition.oracle import ANSWER_STATUSES
from grimore.cognition.answer_protocol import PROTOCOL_VERSION, technical_failure
from grimore.cognition.reembed import reembed_note
from grimore.ingest.extraction_profile import pdf_extraction_profile
from grimore.ingest.parser import MarkdownParser, extraction_diagnostics
from grimore.session import Session
from grimore.utils.config import load_config

DEFAULT_CASES = REPO_ROOT / "eval/phase4_difficult_answers.yaml"
DEFAULT_PDF = REPO_ROOT / "output/pdf/phase4_difficult_humedal.pdf"
LOCAL_HOST = "http://127.0.0.1:11434"
MEASURED_CODE = (
    "eval/benchmark_answers.py", "eval/benchmark_extraction.py",
    "eval/context_order_control.py",
    "grimore/session.py", "grimore/utils/config.py", "grimore/utils/http.py",
    "grimore/utils/security.py", "grimore/utils/ocr.py",
    "grimore/cognition/eval.py", "grimore/cognition/oracle.py",
    "grimore/cognition/sufficiency.py", "grimore/cognition/prompts/sufficiency.txt",
    "grimore/cognition/answer_protocol.py", "grimore/cognition/prompts/cited_answer.txt",
    "grimore/cognition/prompts/oracle.txt", "grimore/cognition/embedder.py",
    "grimore/cognition/connector.py", "grimore/cognition/chunker.py",
    "grimore/cognition/reembed.py", "grimore/cognition/llm_router.py",
    "grimore/cognition/llm_backends/ollama.py",
    "grimore/ingest/parser.py", "grimore/ingest/adapters/base.py",
    "grimore/ingest/adapters/pdf.py", "grimore/ingest/pdf_layout.py",
    "grimore/ingest/extraction_profile.py", "grimore/memory/db.py",
    "grimore/memory/chunks.py", "grimore/memory/notes.py", "grimore/memory/schema.py",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprints(cases: Path, pdf: Path, config: Path) -> dict:
    return {"cases_sha256": sha256(cases), "pdf_sha256": sha256(pdf),
            "config_file_sha256": sha256(config),
            "code_sha256": {p: sha256(REPO_ROOT / p) for p in MEASURED_CODE}}


def negative_control(turns) -> dict:
    """Heuristic only, with no credit for empty/silent/no-context responses.

    It leaves the existing abstention_rate as defined, does not read the
    status as a semantic verdict, and never counts a transport failure as a
    successful abstention. The exact responses and contexts remain in the
    human review queue.
    """
    observations = []
    for turn in turns:
        if not turn.negative:
            continue
        text = turn.answer.strip()
        has_context = any(
            isinstance(d, dict) and isinstance(d.get("passage"), dict)
            and str(d["passage"].get("text") or "").strip()
            for d in turn.source_details
        )
        eligible = bool(
            turn.generated and text and has_context
            and turn.answer_status in ANSWER_STATUSES
            and text not in ("The Oracle is silent.",
                             "Your vault seems empty of relevant whispers on this subject.")
            and not technical_failure(turn.sufficiency)
        )
        observations.append({
            "case_id": turn.case_id, "turn_index": turn.turn_index,
            "delivered_context_nonempty": bool(has_context),
            "eligible_for_cue_check": eligible,
            "abstention_detected": turn.abstained if eligible else None,
            "abstention_verdict": "pending",
        })
    assessed = [o for o in observations if o["eligible_for_cue_check"]]
    detected = sum(o["abstention_detected"] is True for o in assessed)
    return {
        "negative_turns_n": len(observations), "cue_assessed_n": len(assessed),
        "unassessed_n": len(observations) - len(assessed),
        "abstention_cue_hits": detected,
        "abstention_cue_rate": detected / len(assessed) if assessed else None,
        "semantic_abstention_rate": None, "human_review": "pending",
        "observations": observations,
    }


@contextmanager
def local_endpoint():
    previous = os.environ.get("OLLAMA_HOST")
    os.environ["OLLAMA_HOST"] = LOCAL_HOST
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = previous


def installed_identity(http, names: list[str]) -> dict:
    response = http.get(f"{LOCAL_HOST}/api/tags", timeout=10)
    response.raise_for_status()
    models = response.json().get("models", [])
    selected = {}
    for name in names:
        canonical = name if ":" in name else f"{name}:latest"
        match = next((m for m in models if m.get("name") == canonical), None)
        if not match or not match.get("digest"):
            raise ValueError(f"Installed model with digest required: {canonical}; no download attempted")
        # Only the requested models are exported, reduced to name, digest, size
        # and a few public details; other installed models and the private
        # parent-model paths in Ollama's response are left out of the report.
        details = match.get("details") or {}
        selected[name] = {"name": match["name"], "digest": match["digest"],
                          "size_bytes": match.get("size"),
                          "details": {k: details.get(k) for k in
                                      ("format", "family", "parameter_size", "quantization_level")}}
    return selected


def benchmark(cases_path: Path, pdf: Path, config_path: Path, *, model: str | None = None,
              embedding_model: str | None = None, top_k: int = 5, retrieval_k: int = 10,
              context_order: str = "ranked", sufficiency_check: bool = False,
              sufficiency_timeout_s: int = 180) -> dict:
    if top_k < 1 or retrieval_k < top_k:
        raise ValueError("Require retrieval_k >= top_k >= 1")
    if context_order not in STRATEGIES:
        raise ValueError(f"Unknown context-order strategy: {context_order}")
    if type(sufficiency_timeout_s) is not int or sufficiency_timeout_s < 1:
        raise ValueError("Require a positive integer sufficiency timeout")
    cases_path, pdf, config_path = (p.resolve() for p in (cases_path, pdf, config_path))
    cases = load_golden(cases_path)
    if not cases or any(c.split != "tuning" or c.document_formats != ["pdf"] for c in cases):
        raise ValueError("This isolated control accepts only PDF tuning cases, never held-out evaluation")
    before = fingerprints(cases_path, pdf, config_path)
    config = load_config(str(config_path))
    configured_model = config.cognition.model_llm_local
    config.cognition.model_llm_local = model or configured_model
    config.cognition.model_embeddings_local = embedding_model or config.cognition.model_embeddings_local
    config.cognition.llm_backend, config.cognition.allow_remote = "ollama", False
    config.cognition.rerank = False
    config.cognition.answer_sufficiency_check = sufficiency_check
    config.cognition.answer_sufficiency_timeout_s = sufficiency_timeout_s
    config.output.dry_run, config.output.auto_commit = True, False
    config.ingest.pdf_engine, config.ingest.ocr, config.ingest.ocr_language = "pdfplumber", True, "spa"
    config.ingest.ocr_max_pages = 0
    # Pin chunking to the ordinary markdown/section route; no semantic model
    # calls or synthetic retrieval. Optional context-order perturbation acts
    # only AFTER the ordinary ranking, top_k selection and budget acceptance.
    config.ingest.chunker = "markdown"
    with tempfile.TemporaryDirectory(prefix="grimore-phase4-answers-") as temporary, local_endpoint():
        root = Path(temporary)
        vault = root / "vault"
        vault.mkdir()
        copied_pdf = vault / pdf.name
        shutil.copyfile(pdf, copied_pdf)
        config.vault.path, config.vault.formats = str(vault), ["pdf"]
        config.memory.db_path = str(root / "index.db")
        with Session(config) as session:
            models = installed_identity(session.embedder.session, [
                config.cognition.model_llm_local, config.cognition.model_embeddings_local,
            ])
            response = session.embedder.session.get(f"{LOCAL_HOST}/api/version", timeout=10)
            response.raise_for_status()
            server_version = response.json().get("version")
            started = time.monotonic()
            note = MarkdownParser().parse_file(copied_pdf, vault_root=vault, config=config)
            extraction = extraction_diagnostics(note)
            if not note.content.strip() or extraction is None or extraction["status"] == "partial":
                raise ValueError("Complete/recovered PDF extraction required before model measurement")
            chunks = build_candidate_chunks(note, note.content, session.embedder, config)
            if not chunks:
                raise ValueError("No chunks to index")
            note_id = session.db.upsert_note(str(copied_pdf), note.title, "", format="pdf")
            if note_id is None:
                raise ValueError("Could not create isolated document record")
            indexed = reembed_note(session.db, session.embedder, note_id, chunks,
                                  text_truncation=config.cognition.chunk_store_chars)
            if indexed.stored != len(chunks):
                raise ValueError("Incomplete embedding/indexing; no answer benchmark published")
            session.db.complete_note_indexing(
                note_id, note.content_hash, note.file_hash, copied_pdf.stat().st_size,
                copied_pdf.stat().st_mtime_ns, extraction=extraction,
                extraction_profile=pdf_extraction_profile(config),
            )
            index_seconds = time.monotonic() - started
            print(f"Indexed {len(chunks)} chunks; evaluating {len(cases)} cases", file=sys.stderr, flush=True)
            with presentation_control(session.oracle, context_order) as order_audits:
                report = run_eval(session, cases, top_k=top_k, retrieval_k=retrieval_k,
                                  generate=True, judge=False)
            # run_eval records generation errors as failed turns. A failed
            # perturbation must not masquerade as a completed order experiment.
            if len(order_audits) != len(report.turns):
                raise ValueError("Incomplete presentation control; no order benchmark published")
            report.provenance = run_provenance(session, cases_path)
            result = report.summary()
            result["controlled_run"] = {
                "version": 3, "timestamp": datetime.now(timezone.utc).isoformat(),
                "corpus_kind": "synthetic-control", "split": "tuning",
                "isolated_vault_and_index": True, "configured_chat_model": configured_model,
                "endpoint": LOCAL_HOST, "ollama_version": server_version, "models": models,
                "ingest": asdict(config.ingest), "cognition": asdict(config.cognition),
                "client_hardware": hardware_profile("local-answer-control"),
                "inference_hardware": "not independently verified",
                "memory": {"client_peak_rss_mb": _peak_rss_mb(),
                           "child_peak_rss_mb": _peak_rss_mb(children=True),
                           "inference_peak_rss_mb": None,
                           "scope": "process-lifetime peaks; fresh process required; excludes Ollama; do not sum peaks"},
                "inputs": before, "inputs_unchanged": before == fingerprints(cases_path, pdf, config_path),
                "copied_pdf_unchanged": sha256(copied_pdf) == before["pdf_sha256"],
                "corpus": {"source": str(pdf), "title": note.title, "documents": 1,
                           "chunks": len(chunks), "extraction": extraction},
                "index_seconds": index_seconds, "retrieval_k": retrieval_k,
                "generation_samples_per_case": 1, "judge_enabled": False,
                "sufficiency_control": {
                    "enabled": sufficiency_check, "timeout_s": sufficiency_timeout_s,
                    "answer_protocol_version": PROTOCOL_VERSION if sufficiency_check else None,
                    "calls": [{"case_id": t.case_id, "turn_index": t.turn_index,
                               "assessment": t.sufficiency} for t in report.turns],
                    "semantic_support": "unverified",
                    "answer_calls": sum(
                        (t.sufficiency["generation"].get("model_calls", 0)
                         if t.sufficiency and isinstance(t.sufficiency.get("generation"), dict)
                         else int(t.timings.get("generate_s", 0) > 0)) for t in report.turns),
                },
                "case_counts": {"positive_turns": result["positives_n"],
                                "negative_turns": result["negatives_n"]},
                "negative_control": negative_control(report.turns),
                "presentation_control": {
                    "strategy": context_order, "retrieval_ranking_changed": False,
                    "applied_after_context_budget": True,
                    "renumber_refs_in_presentation_order": context_order != "ranked",
                    "calls": order_audits,
                },
                "limitations": ["One controlled PDF; explicit tuning manifest, not a representative corpus",
                                "No distractor documents; absence claims are limited to this isolated PDF",
                                "Abstention cues and model-declared statuses are not semantic abstention verdicts",
                                "Empty, silent, unassessed or no-context responses receive no negative cue-check credit",
                                "Optional context rotation tests passage position, not corpus diversity or reranking",
                                "Literal passage/page coverage is not semantic claim support",
                                "Human claim support remains pending; not a pilot or hardware comparison"],
            }
            return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "grimore.toml")
    parser.add_argument("--model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument("--sufficiency-check", action="store_true",
                        help="Opt-in pre-generation assessment and evidence selection; extra model call")
    parser.add_argument("--sufficiency-timeout-s", type=int, default=180)
    parser.add_argument("--context-order", choices=STRATEGIES, default="ranked",
                        help="Explicit presentation-only control; does not change retrieval ranking")
    parser.add_argument("--output", type=Path, help="Save a complete JSON report; refuse to overwrite")
    args = parser.parse_args()
    try:
        if args.output is not None and args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        with redirect_stdout(sys.stderr):
            result = benchmark(args.cases, args.pdf, args.config, model=args.model,
                               embedding_model=args.embedding_model, top_k=args.top_k,
                               retrieval_k=args.retrieval_k, context_order=args.context_order,
                               sufficiency_check=args.sufficiency_check,
                               sufficiency_timeout_s=args.sufficiency_timeout_s)
        if args.output is not None:
            # An explicit export is an artifact of measurement, not a write
            # to config, corpus or user index. Exclusive creation is mandatory.
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    except Exception as exc:
        print(f"Answer control failed: {exc}", file=sys.stderr)
        return 2
    if args.output is None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Saved report: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
