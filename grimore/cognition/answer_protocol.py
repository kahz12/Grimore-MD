"""Explicit model-selected citations, not citation repair or semantic judging."""
import hashlib
import json
import re
import unicodedata

PROTOCOL_VERSION = 1
RESPONSE_LIMIT = 32_000
STATUSES = frozenset({"supported", "uncertain", "conflicting_evidence", "insufficient_evidence"})


def response_snapshot(response):
    """Bounded, untrusted parsed-model output for review, never a new prompt.

    This is the router's parsed value, not the original HTTP/JSON wire bytes.
    Preserve invalid evidence too. Large outputs retain a prefix and digest,
    explicitly marked incomplete rather than masquerading as a full record.
    """
    try:
        encoded = json.dumps(response, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return {"value": None, "truncated": False, "unavailable": True}
    truncated = len(encoded) > RESPONSE_LIMIT
    return {"value": None if truncated else json.loads(encoded),
            "json_prefix": encoded[:RESPONSE_LIMIT], "chars": len(encoded),
            "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "truncated": truncated, "unavailable": False}


def render_claims(response, details):
    """Validate every claim/ref BEFORE displaying anything; all-or-nothing.

    Each reference must be explicitly supplied by the answer model and belong
    to its delivered context. A checker's earlier choice never supplies a
    missing answer reference. Rendering tokens does not prove claim support.
    """
    if (not isinstance(response, dict) or set(response) != {"claims", "status"}
            or not isinstance(response["status"], str) or response["status"] not in STATUSES
            or not isinstance(response["claims"], list)):
        return "", "uncertain", "invalid_answer_schema"
    claims, status = response["claims"], response["status"]
    if status == "insufficient_evidence":
        return ("", status, None) if not claims else ("", "uncertain", "contradictory_answer")
    if not 1 <= len(claims) <= 64:
        return "", "uncertain", "missing_or_excess_claims"
    available = [d.get("citation_ref") for d in details]
    if (not available or any(not isinstance(ref, str) or not re.fullmatch(r"cite:[1-9]\d*", ref)
                             for ref in available) or len(set(available)) != len(available)):
        return "", "uncertain", "invalid_answer_reference_map"
    available = set(available)
    paragraphs = []
    size = 0
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"text", "refs"}:
            return "", "uncertain", "invalid_claim_schema"
        text, refs = claim["text"], claim["refs"]
        if (not isinstance(text, str) or not text.strip()
                or not any(not c.isspace() and not unicodedata.category(c).startswith("P") for c in text)
                or re.search(r"\[\[|\]\]|\[cite:", text, re.IGNORECASE)):
            return "", "uncertain", "invalid_claim_text"
        if (not isinstance(refs, list) or not refs or len(refs) > len(available)
                or any(not isinstance(ref, str) or ref not in available for ref in refs)
                or len(set(refs)) != len(refs)):
            return "", "uncertain", "missing_or_invalid_claim_refs"
        size += len(text)
        if size > 16_000:
            return "", "uncertain", "answer_exceeds_budget"
        paragraphs.append(text.strip() + " " + " ".join(f"[{ref}]" for ref in refs))
    return "\n\n".join(paragraphs), status, None


def technical_failure(assessment):
    """No technical failure may receive negative-abstention credit."""
    return bool(assessment and (assessment.get("decision") == "unassessed"
                or (assessment.get("generation") or {}).get("decision") == "unassessed"))
