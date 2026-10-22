"""Explicit, temporary presentation perturbation, never a retrieval policy.

Reconstruct whole trusted blocks from delivered records and verify equality
before permuting. Do not split on headers contained in untrusted documents,
consult expected evidence, promote deeper ranking candidates or change budgets.
"""
from contextlib import contextmanager
import hashlib
import re

from grimore.cognition.oracle import _CONTEXT_SEPARATOR, _coverage_context_note
from grimore.utils.security import SecurityGuard

STRATEGIES = ("ranked", "rotate-first-last")


def context_blocks(details: list[dict]) -> list[str]:
    refs = [detail.get("citation_ref") for detail in details]
    if any(not isinstance(ref, str) or not re.fullmatch(r"cite:[1-9]\d*", ref) for ref in refs):
        raise ValueError("Context-order control requires valid short references")
    if len(set(refs)) != len(refs):
        raise ValueError("Context-order control refuses duplicate reference maps")
    blocks = []
    for detail in details:
        if detail.get("source_state") == "provided":
            raise ValueError("Context-order control does not rearrange pinned sources")
        label = detail.get("label")
        text = (detail.get("passage") or {}).get("text")
        if not isinstance(label, str) or not isinstance(text, str) or not text:
            raise ValueError("Context-order control requires complete delivered blocks")
        safe_text = SecurityGuard.wrap_untrusted(SecurityGuard.sanitize_prompt(text), label="source")
        blocks.append(f"Citation token: [{detail['citation_ref']}]\n"
                      f"--- Source: [[{label}]] ---\n{_coverage_context_note(detail)}{safe_text}")
    return blocks


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def reorder_payload(payload, strategy: str):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown context-order strategy: {strategy}")
    context, sources, retrieved, details = payload
    audit = {"strategy": strategy, "original_context_indices": list(range(len(details))),
             "presented_original_indices": list(range(len(details))), "changed": False,
             "context_chars_before": len(context or ""), "context_chars_after": len(context or ""),
             "context_sha256_before": _digest(context or ""),
             "context_sha256_after": _digest(context or ""), "reconstruction_verified": None}
    if strategy == "ranked" or context is None or len(details) < 2:
        return payload, audit
    original_blocks = context_blocks(details)
    if _CONTEXT_SEPARATOR.join(original_blocks) != context:
        raise ValueError("Context reconstruction mismatch; refuse to alter generation")
    # Fixed positional permutation, independent of question/expected answer.
    # Reuse the original safe token set in presentation order, not the old
    # token attached to each record: the former first passage is no longer cite:1.
    order = list(range(1, len(details))) + [0]
    refs = [detail["citation_ref"] for detail in details]
    presented = [{**details[index], "citation_ref": refs[position]}
                 for position, index in enumerate(order)]
    reordered = _CONTEXT_SEPARATOR.join(context_blocks(presented))
    if len(reordered) != len(context):
        raise ValueError("Context-order control must preserve the accepted character budget")
    audit.update({"presented_original_indices": order, "changed": True,
                  "context_chars_after": len(reordered), "context_sha256_after": _digest(reordered),
                  "reconstruction_verified": True,
                  "passages": [{"original_context_index": index,
                                "presented_context_index": position,
                                "id": details[index].get("id"), "label": details[index]["label"],
                                "original_ref": details[index]["citation_ref"],
                                "presented_ref": presented[position]["citation_ref"],
                                "passage_sha256": _digest(details[index]["passage"]["text"])}
                               for position, index in enumerate(order)]})
    # Ranking pool is the actual connector result. Citation-facing labels are
    # reordered along with delivered records, not rescored as a retrieval result.
    return (reordered, list(dict.fromkeys(d["label"] for d in presented)), retrieved, presented), audit


@contextmanager
def presentation_control(oracle, strategy: str):
    """Instrument only this isolated Oracle; restore even after exceptions."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown context-order strategy: {strategy}")
    original = oracle._build_context_payload
    had_override = "_build_context_payload" in vars(oracle)
    previous_override = vars(oracle).get("_build_context_payload")
    audits = []

    def build(*args, **kwargs):
        payload, audit = reorder_payload(original(*args, **kwargs), strategy)
        audit["question"] = args[0] if args else kwargs.get("question")
        audits.append(audit)
        return payload

    oracle._build_context_payload = build
    try:
        yield audits
    finally:
        if had_override:
            oracle._build_context_payload = previous_override
        else:
            del oracle._build_context_payload
