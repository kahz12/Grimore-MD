"""Read-only aggregation of maintenance signals for human review."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from grimore.cognition.chronicler import Chronicler
from grimore.cognition.dedupe import find_exact_duplicates, find_near_duplicates
from grimore.cognition.embedder import Embedder
from grimore.cognition.mirror import Mirror


_KIND_ORDER = {
    "contradiction": 0,
    "duplicate_exact": 1,
    "duplicate_near": 2,
    "review_due": 3,
}
_SQL_BATCH = 900


def _batched(values: list[int] | list[str], size: int = _SQL_BATCH):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _document(note_id: int | None, path: str, title: str = "") -> dict:
    return {
        "note_id": note_id,
        "path": path,
        "title": title or Path(path).stem,
    }


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _note_metadata_for_paths(db, paths: set[str]) -> dict[str, tuple[int, str]]:
    if not paths:
        return {}
    out: dict[str, tuple[int, str]] = {}
    ordered = sorted(paths)
    with db._get_connection() as conn:
        for batch in _batched(ordered):
            marks = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT id, path, title FROM notes WHERE path IN ({marks})",
                tuple(batch),
            ).fetchall()
            out.update({path: (int(note_id), title or "") for note_id, path, title in rows})
    return out


def _chunk_summary(db, note_ids: set[int]) -> tuple[dict[int, int], dict[int, int]]:
    """Return per-note chunk counts and the first embedding id."""
    counts: dict[int, int] = {}
    first_ids: dict[int, int] = {}
    ordered = sorted(note_ids)
    if not ordered:
        return counts, first_ids
    with db._get_connection() as conn:
        for batch in _batched(ordered):
            marks = ",".join("?" * len(batch))
            rows = conn.execute(
                "SELECT note_id, COUNT(*), MIN(id) FROM embeddings "
                f"WHERE note_id IN ({marks}) GROUP BY note_id",
                tuple(batch),
            ).fetchall()
            for note_id, count, first_id in rows:
                counts[int(note_id)] = int(count)
                first_ids[int(note_id)] = int(first_id)
    return counts, first_ids


def _near_vectors(db, note_ids: set[int]) -> dict[int, list[tuple[int, list[float]]]]:
    vectors: dict[int, list[tuple[int, list[float]]]] = {}
    ordered = sorted(note_ids)
    if not ordered:
        return vectors
    with db._get_connection() as conn:
        for batch in _batched(ordered):
            marks = ",".join("?" * len(batch))
            rows = conn.execute(
                "SELECT id, note_id, vector FROM embeddings "
                f"WHERE note_id IN ({marks}) AND vector IS NOT NULL ORDER BY id",
                tuple(batch),
            ).fetchall()
            for embedding_id, note_id, blob in rows:
                try:
                    vector = Embedder.deserialize_vector(blob)
                except Exception:
                    continue
                vectors.setdefault(int(note_id), []).append((int(embedding_id), vector))
    return vectors


def _mean_vector(rows: list[tuple[int, list[float]]]) -> list[float]:
    if not rows:
        return []
    dimension = len(rows[0][1])
    compatible = [vector for _embedding_id, vector in rows if len(vector) == dimension]
    if not compatible:
        return []
    mean = [sum(vector[i] for vector in compatible) / len(compatible) for i in range(dimension)]
    return Embedder.normalize(mean)


def _near_passage_ids(
    vectors: dict[int, list[tuple[int, list[float]]]],
    a_id: int,
    b_id: int,
) -> list[int]:
    """Choose the passage in each note closest to the other note's mean."""
    a_rows = vectors.get(a_id, [])
    b_rows = vectors.get(b_id, [])
    a_mean = _mean_vector(a_rows)
    b_mean = _mean_vector(b_rows)
    if not a_mean or not b_mean or len(a_mean) != len(b_mean):
        return []

    def best(rows: list[tuple[int, list[float]]], target: list[float]) -> int | None:
        compatible = [(eid, vector) for eid, vector in rows if len(vector) == len(target)]
        if not compatible:
            return None
        return max(
            compatible,
            key=lambda row: (Embedder.dot_product(row[1], target), -row[0]),
        )[0]

    return [eid for eid in (best(a_rows, b_mean), best(b_rows, a_mean)) if eid is not None]


def _indexed_passage(record: dict, *, role: str) -> dict | None:
    text = record.get("full_text") or record.get("excerpt") or ""
    if not text.strip():
        return None
    return {
        "note_id": record["note_id"],
        "path": record["path"],
        "title": record["title"] or Path(record["path"]).stem,
        "text": text,
        "role": role,
        "source": "indexed_chunk",
        "source_id": (
            f"note:{record['note_id']}:chunk:{record['chunk_index']}:"
            f"embedding:{record['embedding_id']}"
        ),
        "indexed_file_hash": record.get("file_hash"),
        "chunk_index": record["chunk_index"],
        "page": record.get("page"),
        "heading": record.get("heading"),
        "complete": record.get("full_text") is not None,
    }


def _claim_passage(
    document: dict,
    text: str,
    char_start: int | None,
    char_end: int | None,
) -> dict:
    return {
        **document,
        "text": text,
        "role": "contradicting_claim",
        "source": "extracted_claim",
        "char_start": char_start,
        "char_end": char_end,
        "complete": True,
    }


def _coverage(
    *,
    status: str,
    basis: str,
    documents_analyzed: int,
    documents_total: int,
    passages_shown: int,
    passages_available: int,
    chunks_analyzed: int | None,
    limitations: list[str],
    caps: dict | None = None,
) -> dict:
    return {
        "status": status,
        "basis": basis,
        "documents": {"analyzed": documents_analyzed, "total": documents_total},
        "passages": {"shown": passages_shown, "available": passages_available},
        "chunks_analyzed": chunks_analyzed,
        "limitations": limitations,
        "caps": caps or {},
    }


def build_review_inbox(
    session,
    *,
    duplicate_limit: int = 30,
    include_inactive: bool = False,
) -> dict:
    """Return inspectable review candidates without changing persisted state.

    Contradictions come only from already persisted open Mirror findings;
    duplicate checks are deterministic and LLM-free; review-due notes use
    Chronicler's non-mutating snapshot path. Evidence is drawn from stored
    claims and indexed chunks, so this GET never invokes a model.
    """
    mirror = Mirror(session)
    contradictions = mirror.list_open()
    exact_groups = find_exact_duplicates(session.db)
    threshold = float(getattr(session.config.cognition, "dedupe_threshold", 0.90))
    near_pairs = find_near_duplicates(
        session.db, threshold=threshold, limit=duplicate_limit,
    )
    stale_notes = Chronicler(session).list_stale_snapshot()

    paths = {
        path
        for row in contradictions
        for path in (row.note_a, row.note_b)
    } | {note.path for note in stale_notes}
    metadata_by_path = _note_metadata_for_paths(session.db, paths)

    note_ids = {
        note_id
        for group in exact_groups
        for note_id, _path, _title in group.notes
    } | {
        note_id
        for pair in near_pairs
        for note_id in (pair.a_id, pair.b_id)
    } | {
        metadata_by_path[note.path][0]
        for note in stale_notes
        if note.path in metadata_by_path
    } | {
        note_id for note_id, _title in metadata_by_path.values()
    }
    chunk_counts, first_chunk_ids = _chunk_summary(session.db, note_ids)
    near_note_ids = {
        note_id
        for pair in near_pairs
        for note_id in (pair.a_id, pair.b_id)
    }
    vectors = _near_vectors(session.db, near_note_ids)
    near_ids = {
        embedding_id
        for pair in near_pairs
        for embedding_id in _near_passage_ids(vectors, pair.a_id, pair.b_id)
    }
    selected_ids = set(first_chunk_ids.values()) | near_ids
    evidence_records = session.db.get_evidence_records(selected_ids)
    classifications = session.db.get_note_classifications(note_ids)
    timelines = session.db.get_note_timelines(note_ids)

    items: list[dict] = []
    for row in contradictions:
        meta_a = metadata_by_path.get(row.note_a)
        meta_b = metadata_by_path.get(row.note_b)
        doc_a = _document(meta_a[0] if meta_a else None, row.note_a, meta_a[1] if meta_a else "")
        doc_b = _document(meta_b[0] if meta_b else None, row.note_b, meta_b[1] if meta_b else "")
        raw = session.db.get_contradiction(row.id)
        cs_a = raw[12] if raw else None
        ce_a = raw[13] if raw else None
        cs_b = raw[14] if raw else None
        ce_b = raw[15] if raw else None
        passages = [
            _claim_passage(doc_a, row.claim_a, cs_a, ce_a),
            _claim_passage(doc_b, row.claim_b, cs_b, ce_b),
        ]
        items.append({
            "id": f"contradiction:{row.id}",
            "kind": "contradiction",
            "status": "candidate",
            "priority": row.severity,
            "detected_at": row.detected_at,
            "documents": [doc_a, doc_b],
            "reason": row.explanation,
            "passages": passages,
            "coverage": _coverage(
                status="partial",
                basis="extracted_claim_pair",
                documents_analyzed=2,
                documents_total=2,
                passages_shown=2,
                passages_available=2,
                chunks_analyzed=None,
                limitations=[
                    "claim_pair_only", "claim_extraction_bounded", "model_candidate",
                ],
                caps={
                    "max_sections_per_document": session.config.cognition.claim_max_sections,
                    "max_claims_per_document": (
                        session.config.cognition.claim_max_sections
                        * session.config.cognition.claim_max_claims_per_section
                    ),
                    "max_characters_per_document": (
                        session.config.cognition.claim_max_sections
                        * session.config.cognition.claim_max_chars
                    ),
                },
            ),
            "metadata": {"contradiction_id": row.id, "severity": row.severity},
        })

    for group in exact_groups:
        stable = sha256(group.content_hash.encode("utf-8")).hexdigest()[:16]
        documents = [_document(note_id, path, title) for note_id, path, title in group.notes]
        passages = []
        for note_id, _path, _title in group.notes:
            record = evidence_records.get(first_chunk_ids.get(note_id, -1))
            passage = _indexed_passage(record, role="representative") if record else None
            if passage:
                passages.append(passage)
        available = sum(chunk_counts.get(note_id, 0) for note_id, _path, _title in group.notes)
        items.append({
            "id": f"duplicate-exact:{stable}",
            "kind": "duplicate_exact",
            "status": "candidate",
            "priority": "high",
            "detected_at": None,
            "documents": documents,
            "reason": "matching normalized content hash",
            "passages": passages,
            "coverage": _coverage(
                status="complete",
                basis="normalized_content_hash",
                documents_analyzed=len(documents),
                documents_total=len(documents),
                passages_shown=len(passages),
                passages_available=available,
                chunks_analyzed=None,
                limitations=["representative_passages_only", "metadata_ignored"],
            ),
            "metadata": {"copies": len(group.notes)},
        })

    for pair in near_pairs:
        documents = [
            _document(pair.a_id, pair.a_path, pair.a_title),
            _document(pair.b_id, pair.b_path, pair.b_title),
        ]
        passage_ids = _near_passage_ids(vectors, pair.a_id, pair.b_id)
        passages = []
        for embedding_id in passage_ids:
            record = evidence_records.get(embedding_id)
            passage = _indexed_passage(record, role="similarity_evidence") if record else None
            if passage:
                passages.append(passage)
        total_chunks = chunk_counts.get(pair.a_id, 0) + chunk_counts.get(pair.b_id, 0)
        items.append({
            "id": f"duplicate-near:{pair.a_id}:{pair.b_id}",
            "kind": "duplicate_near",
            "status": "candidate",
            "priority": "medium",
            "detected_at": None,
            "documents": documents,
            "reason": "mean embedding similarity above the configured threshold",
            "passages": passages,
            "coverage": _coverage(
                status="complete" if len(passages) == 2 else "partial",
                basis="mean_chunk_embeddings",
                documents_analyzed=2,
                documents_total=2,
                passages_shown=len(passages),
                passages_available=total_chunks,
                chunks_analyzed=total_chunks,
                limitations=["semantic_similarity_not_identity", "result_limit_applied"],
                caps={"max_results": duplicate_limit},
            ),
            "metadata": {
                "score": round(float(pair.score), 6),
                "threshold": threshold,
                "result_limit": duplicate_limit,
            },
        })

    for note in stale_notes:
        stable = sha256(note.path.encode("utf-8")).hexdigest()[:16]
        note_meta = metadata_by_path.get(note.path)
        note_id = note_meta[0] if note_meta else None
        document = _document(note_id, note.path, note.title)
        record = evidence_records.get(first_chunk_ids.get(note_id, -1)) if note_id else None
        passage = _indexed_passage(record, role="representative") if record else None
        passages = [passage] if passage else []
        available = chunk_counts.get(note_id, 0) if note_id else 0
        items.append({
            "id": f"review-due:{stable}",
            "kind": "review_due",
            "status": "candidate",
            "priority": "high" if note.likely_stale else "medium",
            "detected_at": None,
            "documents": [document],
            "reason": "category review window exceeded",
            "passages": passages,
            "coverage": _coverage(
                status="partial",
                basis="category_review_window",
                documents_analyzed=1,
                documents_total=1,
                passages_shown=len(passages),
                passages_available=available,
                chunks_analyzed=0,
                limitations=["age_not_content_validity", "representative_passages_only"],
            ),
            "metadata": {
                "category": note.category,
                "last_verified": note.last_verified,
                "window_days": note.window_days,
                "days_overdue": note.days_overdue,
                "likely_stale": note.likely_stale,
            },
        })

    for item in items:
        for document in item["documents"]:
            classification = classifications.get(document.get("note_id"))
            if classification:
                document["classification"] = classification
            timeline = timelines.get(document.get("note_id"))
            if timeline:
                document["timeline"] = timeline

    items.sort(key=lambda item: (
        _KIND_ORDER.get(item["kind"], 99),
        {"high": 0, "medium": 1, "low": 2}.get(item["priority"], 3),
        item["id"],
    ))
    states = session.db.get_review_candidate_states(item["id"] for item in items)
    now = datetime.now(timezone.utc)
    visible_items = []
    for item in items:
        state = states.get(item["id"])
        if state:
            item["review_state"] = {
                key: state.get(key)
                for key in (
                    "decision", "decided_at", "postpone_until", "source",
                )
            }
        inactive = bool(state and state.get("decision") in {"confirm", "dismiss"})
        if state and state.get("decision") == "postpone":
            postpone_until = _parse_utc(state.get("postpone_until"))
            inactive = postpone_until is not None and postpone_until > now
            item["review_state"]["postpone_active"] = inactive
        if include_inactive or not inactive:
            visible_items.append(item)

    counts = dict.fromkeys(_KIND_ORDER, 0)
    for item in visible_items:
        counts[item["kind"]] += 1
    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "candidates_only",
        "counts": {"all": len(visible_items), **counts},
        "items": visible_items,
        "history": session.db.list_review_decisions(limit=50),
    }
