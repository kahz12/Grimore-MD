"""Refresh PDF chunks when source bytes change but extracted text does not."""
from __future__ import annotations

from grimore.cognition.chunker import build_candidate_chunks
from grimore.cognition.reembed import ReembedResult, reembed_note
from grimore.ingest.extraction_profile import pdf_index_profile
from grimore.ingest.parser import extraction_diagnostics


def reconcile_unchanged_pdf(
    db,
    embedder,
    note,
    config,
    *,
    clean_content: str,
    file_hash: str,
    size_bytes: int,
    mtime_ns: int,
    document_dates: dict,
) -> ReembedResult:
    """Refresh chunk anchors/content, then publish the new file fingerprint.

    ``reembed_note`` reuses vectors for identical chunk text and only updates
    anchors. If page boundaries changed enough to alter chunk text, it embeds
    those changed chunks. A failed or incomplete run leaves the old file
    fingerprint in place so the next scan retries.
    """
    if note.format != "pdf":
        raise ValueError("PDF reconciliation requires a PDF note")
    note_id = db.get_note_id_by_path(str(note.path))
    if note_id is None:
        raise ValueError(f"indexed PDF not found: {note.path}")
    candidates = build_candidate_chunks(note, clean_content, embedder, config)
    result = reembed_note(
        db, embedder, note_id, candidates,
        text_truncation=getattr(config.cognition, "chunk_store_chars", 500),
    )
    if result.stored != len(candidates):
        raise RuntimeError("embedding incomplete; PDF remains pending for retry")
    extraction = extraction_diagnostics(note)
    db.complete_note_indexing(
        note_id, note.content_hash, file_hash, size_bytes, mtime_ns,
        extraction_profile=pdf_index_profile(config, extraction),
        extraction=extraction,
        **document_dates,
    )
    return result
