"""Optional, fallible pre-generation evidence selection. No expected answers.

Literal quote validation establishes provenance, never semantic entailment.
Unknown/failed decisions cannot authorize factual generation.
"""
import hashlib
from pathlib import Path
import re
import time
import unicodedata

from grimore.utils.security import SecurityGuard
from grimore.cognition.answer_protocol import RESPONSE_LIMIT, response_snapshot


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _literal(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def context_blocks(details, coverage_note):
    """Reconstruct whole blocks, never split on untrusted source headers."""
    refs = set()
    blocks = []
    for detail in details:
        ref, label = detail.get("citation_ref"), detail.get("label")
        text = (detail.get("passage") or {}).get("text")
        if (not isinstance(ref, str) or not re.fullmatch(r"cite:[1-9]\d*", ref)
                or ref in refs or not isinstance(label, str)
                or not isinstance(text, str) or not text):
            raise ValueError("Invalid delivered evidence map")
        refs.add(ref)
        pinned = " (pinned)" if detail.get("source_state") == "provided" else ""
        coverage = "" if pinned else coverage_note(detail)
        safe = SecurityGuard.wrap_untrusted(SecurityGuard.sanitize_prompt(text), label="source")
        blocks.append(f"Citation token: [{ref}]\n--- Source: [[{label}]]{pinned} ---\n{coverage}{safe}")
    return blocks


def validate_decision(response, details):
    if (not isinstance(response, dict) or set(response) != {"answerable", "evidence"}
            or type(response["answerable"]) is not bool
            or not isinstance(response["evidence"], list)):
        return "unassessed", [], "invalid_schema"
    evidence = response["evidence"]
    if not response["answerable"]:
        return ("insufficient", [], None) if not evidence else ("unassessed", [], "contradictory_decision")
    if not evidence or len(evidence) > len(details):
        return "unassessed", [], "missing_or_excess_evidence"
    by_ref = {d["citation_ref"]: (i, d) for i, d in enumerate(details)}
    verified, seen = [], set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"ref", "quote"}:
            return "unassessed", [], "invalid_evidence"
        ref, quote = item["ref"], item["quote"]
        if not isinstance(ref, str) or ref not in by_ref or ref in seen:
            return "unassessed", [], "unknown_or_duplicate_ref"
        index, detail = by_ref[ref]
        # Check what was actually shown after sanitization, not hidden text.
        shown = SecurityGuard.sanitize_prompt(detail["passage"]["text"])
        if (not isinstance(quote, str) or not quote.strip()
                or len(quote) > RESPONSE_LIMIT or _literal(quote) not in _literal(shown)):
            return "unassessed", [], "quote_not_in_passage"
        seen.add(ref)
        verified.append({"ref": ref, "quote": quote, "candidate_detail_index": index,
                         "label": detail["label"], "passage_sha256": digest(shown)})
    return "answerable", verified, None


def assess(router, question, context, details, coverage_note, *, timeout_s=180):
    started = time.monotonic()
    audit = {"decision": "unassessed", "error": None, "quote_evidence": [],
             "version": 2, "model_response": None,
             "candidate_source_details": details, "context_sha256": digest(context),
             "candidate_context_chars": len(context), "answer_generation_allowed": False,
             "model_calls": 0, "semantic_support": "unverified"}
    try:
        blocks = context_blocks(details, coverage_note)
        if not blocks or "\n\n".join(blocks) != context:
            raise ValueError("Context reconstruction mismatch")
        if len(question) > 8000:
            raise ValueError("Question exceeds checker budget")
        template = (Path(__file__).parent / "prompts/sufficiency.txt").read_text(encoding="utf-8")
        safe_question = SecurityGuard.wrap_untrusted(SecurityGuard.sanitize_prompt(question), label="question")
        audit["model_calls"] = 1
        response = router.complete(prompt=f"Assess this question:\n{safe_question}",
                                   system_prompt=template.replace("{context}", context),
                                   timeout_s=timeout_s, optional=True)
        audit["model_response"] = response_snapshot(response)
        decision, evidence, error = validate_decision(response, details)
        declared = response.get("answerable") if isinstance(response, dict) else None
        audit.update(decision=decision, quote_evidence=evidence, error=error,
                     model_answerable=declared if type(declared) is bool else None)
        selected = {e["ref"] for e in evidence}
        # Preserve accepted order, labels, IDs and reference tokens. No deeper
        # candidates, reranking, answer-based repairs or new page guesses.
        chosen = [(d, b) for d, b in zip(details, blocks, strict=True) if d["citation_ref"] in selected]
        selected_context = "\n\n".join(b for _, b in chosen)
        selected_details = [d for d, _ in chosen]
        audit["answer_generation_allowed"] = decision == "answerable"
        audit["selected_context_sha256"] = digest(selected_context)
        audit["selected_context_chars"] = len(selected_context)
        return selected_context, selected_details, audit
    except Exception:
        # Do not expose endpoint exceptions or source contents in error logs.
        audit["error"] = "checker_failed"
        return "", [], audit
    finally:
        audit["elapsed_s"] = time.monotonic() - started


def blocked_answer(question, *, failed=False, stage="assessment"):
    spanish = "¿" in question or bool(re.search(r"\b(qué|cuál|cómo|cuánt\w*|dónde)\b", question.lower()))
    if failed and stage == "generation":
        return ("No se pudo validar una respuesta con referencias; no se muestra una respuesta factual."
                if spanish else "A referenced answer could not be validated; no factual answer is displayed.")
    if failed:
        return ("No se pudo comprobar la evidencia disponible; no se ha generado una respuesta factual."
                if spanish else "Evidence checking failed; no factual answer was generated.")
    return ("Evidencia insuficiente: los pasajes proporcionados no establecen el dato o la relación solicitados."
            if spanish else "I cannot answer from the provided passages: they do not establish the requested fact or relationship.")
