"""Temporary stdio server for the editor-client interoperability test; no models."""
from pathlib import Path
import sys
from unittest.mock import MagicMock

from grimore.cognition.embedder import Embedder
from grimore.cognition.oracle import Oracle
from grimore.mcp_server import run_stdio
from grimore.session import Session
from grimore.utils.config import Config, MemoryConfig, OutputConfig, VaultConfig
from grimore.utils.hashing import sha256_file
from grimore.utils.logger import setup_logger


def main():
    setup_logger(json_format=True, stream=sys.stderr)
    root = Path(sys.argv[1]).resolve()
    session = Session(Config(
        vault=VaultConfig(path=str(root), ignored_dirs=[]),
        memory=MemoryConfig(db_path=str(root.parent / "fixture.db")),
        output=OutputConfig(dry_run=True, auto_commit=False),
    ))
    db = session.db
    note_id = db.upsert_note(str(root / "note.md"), "Evidence", "same", file_hash=sha256_file(root / "note.md"))
    db.upsert_note(str(root / "copy.md"), "Copy", "same", file_hash=sha256_file(root / "copy.md"))
    for index in range(2):
        db.store_embedding(
            note_id, index, "Repeated preview", Embedder.serialize_vector([1.0, 0.0]),
            full_text_content=f"Exact indexed passage {index}", heading="Heading",
        )
    record = list(db.get_evidence_records(key[0] for key in db.get_embedding_keys()).values())[-1]
    oracle = Oracle.__new__(Oracle)
    oracle.config, oracle.db = session.config, db
    oracle.embedder, oracle.router, oracle.connector = MagicMock(), MagicMock(), MagicMock()
    def embed(_query):
        # Even an adapter's stray print must not corrupt the protocol stream.
        print("fixture diagnostic: not a wire frame")
        return [1.0, 0.0]

    oracle.embedder.embed.side_effect = embed
    oracle.connector.find_hybrid.return_value = [{
        "embedding_id": record["embedding_id"], "note_id": note_id,
        "text": "Repeated preview", "score": 0.9,
    }]
    session._oracle, session._embedder = oracle, oracle.embedder
    try:
        run_stdio(session)
    finally:
        assert oracle.router.mock_calls == []
        session.close()


if __name__ == "__main__":
    main()
