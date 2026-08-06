# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Six previously hard-coded constants are now `grimore.toml` keys, all read
  with defaults so existing configs behave identically: `chunk_store_chars`,
  `context_max_chars`, `embed_batch_size`, `circuit_failure_threshold`,
  `circuit_cooldown_s` (`[cognition]`) and `max_turns` (`[shell]`). Setting
  `max_turns = 0` disables conversation memory outright.
- `grimore.toml.example` — a documented template replacing the tracked
  `grimore.toml`, which carried one machine's vault path, model names and
  timeouts. Copy it and edit your copy; the real file is now gitignored.
- `[cognition].vec_matrix_cache` (default on) — the dense scoring matrix is
  stored as a `.npy` beside the database and reloaded memory-mapped, so a
  one-shot `grimore ask` no longer rebuilds it from the whole embeddings table.
  Costs 4 bytes per dimension per chunk on disk (~53 MB for 17,500 chunks at
  768 dims). Setting it false restores the previous behaviour and deletes any
  existing file.
- `bench/` — a benchmark harness for the persistence and retrieval paths:
  a seeded deterministic vault generator, an Ollama-shaped LLM stub, and a
  measurement script that counts SQLite connections and statements as well as
  timing scan / ask / connect / dense-matrix load. `bench/BASELINE_EN.md` and
  `bench/BASELINE_ES.md` record the numbers behind every change below.

### Changed
- `Database` now keeps **one SQLite connection per thread** for its lifetime
  instead of opening and closing one per operation, applying the PRAGMAs and
  the sqlite-vec load once rather than on each of the 73 data-access paths.
  It also enables `busy_timeout`, `cache_size` and `mmap_size`, which a
  connection discarded microseconds later could never benefit from. Measured
  on a 2000-note vault: 66,396 connection opens drop to 1 per thread and scan
  wall-clock falls 87.7%, because at 7.17 ms per connection the WAL `fsync`
  on every commit was the dominant cost of indexing. `Database.close()`
  reclaims connections opened on other threads, and is called from
  `Session.close()` and `daemon.stop()`.
- Embeddings for a note are written in **one transaction** instead of one per
  chunk (`store_embeddings_bulk`), for a further 16.2% off scan on a 600-note
  vault. The single-row `store_embedding` is unchanged.
- The Oracle resolves the titles and citation anchors for a whole result set
  in **two queries instead of two per retrieved chunk** (`get_note_titles`,
  `get_chunk_anchors_bulk`): 10 metadata queries per warm `ask` become 2.
  Retrieval itself is untouched, so answers and citations are byte-identical.
- The dense scoring matrix is built from a vector-only load instead of one
  that also fetched every chunk's stored text and kept it cached. Peak memory
  for that load drops 53.6% (121.2 MB to 56.3 MB on a 2000-note vault), leaving
  2.6 MB of overhead above the matrix itself where there used to be 67.6 MB.
  Chunk text is now fetched for the handful of results that need it, and
  `find_similar_notes(with_text=False)` skips even that for the callers that
  only read note ids and scores.
- `connect` scores every note against every chunk in one blocked matrix
  multiply rather than calling the single-query path once per note: 20.5 s down
  to 3.1 s on 2000 notes, with 2000 fewer queries. Suggested links are
  unchanged. Scores can differ in the last bits, since a matrix-by-matrix
  product accumulates in a different order than matrix-by-vector; measured at
  4.8e-07 worst case, with no observed change of ranking.
- HTTP read timeouts are no longer retried on inference calls. A read timeout
  on `/api/generate` means the model is still working, so retrying discarded
  work in progress and turned one over-budget call into three, reporting the
  failure 3x later than the configured timeout. Connection retries are kept —
  those are what let the daemon start before Ollama is up.
- `migrate-embeddings` now drops the matrix cache on its final swap. The swap
  re-inserts every row under its original id, so the row count and max id come
  out identical even though every vector changed — nothing the cache seals on
  can detect that.
- EN/ES user guides document the new config keys, and explain how to **size**
  `request_timeout_s` / `stream_timeout_s` from a measurement rather than by
  guessing. `stream_timeout_s` in particular has to cover the entire prompt
  eval: Ollama emits no first token until it finishes, so a budget below that
  aborts having received nothing, which surfaces as the Oracle answering
  "returned no tokens" rather than as a timeout.

### Fixed
- The sqlite-vec mirror could break silently. `_create_vec_table` set the
  cached dimension inside the transaction that creates `embeddings_vec`, so a
  rollback undid the DDL but left the attribute pointing at a table that no
  longer existed. Every later write then inserted into nothing, was downgraded
  to a warning, and the scan carried on — `embeddings` kept saving while no
  vector was mirrored, leaving a vec index that answers with a fraction of the
  vault. Recovery now happens within the same call, since the failing batch's
  rows commit either way.

## [3.2.0] - 2026-07-21

### Added
- `serve --strict-token` — requires the bearer token from loopback
  clients too. On Android/Termux any app on the device can reach
  localhost ports, so loopback is not a trust boundary there; strict
  mode closes that gap (the bundled web UI sends no token, so drive the
  API with explicit `Authorization` headers in this mode).
- `dedupe` command — finds duplicate notes with two deterministic, LLM-free
  signals: **exact** (bodies sharing a `content_hash`) and **near** (note
  pairs whose mean chunk vectors exceed a cosine threshold). Report-only —
  it never touches the vault or the index. Flags: `--threshold/-t`,
  `--limit/-n`, `--export/-e`.
- `eval` retrieval-quality harness, substantially expanded: new
  `--retrieval-k`, `--retrieval-only`, `--baseline` (hybrid RRF vs
  dense-only, per-metric delta), `--judge/--no-judge`, `--export`,
  `--history` (JSONL run ledger), `--compare` (regression gate, non-zero
  exit on any drop) and `--json` flags, plus Hit@1 / Hit@3, MRR and recall@k
  metrics. Expected-source matching is token-normalised (accent/emoji/case
  folded), so golden entries stay short and robust.
- CI now runs a mypy type-check job (gated on `grimore.memory` and
  `grimore.utils`) and reports test coverage (`pytest-cov`, report-only).
- `CHANGELOG.md` (this file).

### Changed
- Split the ~1,800-line `memory/db.py` "god object" into nine domain mixins
  (`schema`, `search`, `notes`, `chunks`, `embedding_migration`, `tags`,
  `upkeep`, `freshness`, `mirror_store`) plus a `_base` typing contract.
  `Database`'s public API is unchanged — every caller still imports the same
  class from the same module.
- Hybrid retrieval (RRF fusion of BM25 + dense) tuning in the connector and
  Oracle, with matching test coverage.
- CI actions bumped off the deprecated Node 20 runtime (`checkout@v7`,
  `setup-python@v6`).
- EN/ES user guides updated for the new `eval` flags and `dedupe`.

### Removed
- `requirements.txt` — it was an unpinned duplicate of the dependencies in
  `pyproject.toml`, which is the single source of truth. Install with
  `pip install -e .` (add extras as needed, e.g. `pip install -e ".[serve]"`).

### Fixed
- SQLite connections opened by `Database._get_connection` were never
  explicitly closed, leaking a file descriptor per call — harmless in
  one-shot CLI runs but a steady drip in the long-running daemon. It is now
  a context manager that commits/rolls back **and** always closes.
- Note and sidecar writes raised "a bytes-like object is required, not 'str'"
  on installs that resolved python-frontmatter 1.3.0, whose `dump()` no longer
  encodes when handed a binary file handle. `FrontmatterWriter` now serializes
  with `dumps()` and encodes explicitly, so writes work across frontmatter
  versions.
- Cleared all `ruff` findings (E402 import placement in `daemon.py` /
  `preflight.py`, explicit `strict=` on every `zip()`, an unused loop
  variable), and aligned the CI test job's dependencies with the suite.
- A migration test asserted on `click.exceptions.Exit`; recent `typer` no
  longer installs `click` as a top-level module, so the test now asserts
  `typer.Exit` (the type the code actually raises).

### Security
- Bounded the *decompressed* size of every zip member read from
  `.docx` / `.odt` / `.epub` files (100 MB ceiling, enforced while
  inflating — never trusting the zip header). The existing per-format
  caps bound the archive on disk, but deflate reaches ~1000:1, so a
  few-MB member inside a size-legal file could still balloon to
  gigabytes in memory before any parser saw it (zip-bomb DoS).
- The HTTP API's `GET /api/notes/{id}` and the MCP `grimore_get_note`
  tool now re-assert vault containment on the DB-stored path before
  reading the file, closing the one spot where a tampered index row or
  a symlink swapped after scanning could have exposed a file outside
  the vault to a caller. Escaping paths read as a plain 404 / not-found.
- Failed API-token attempts are now throttled per peer address: after
  10 bad tokens within 60 s, further attempts get HTTP 429 until the
  window expires. The constant-time compare already blocked timing
  attacks; this bounds the online guess *rate* on a LAN bind.
- `serve` warns when the API token is passed as a command-line argument
  (visible to other local users via `ps`) and recommends the
  `GRIMORE_API_TOKEN` env var; the guides' LAN examples now use the env
  var alone.

## [3.1.0] - 2026-06-24

First tagged release. Earlier history lives in the git log; this entry
summarizes the capabilities present at this version.

### Added
- Multi-format ingest: Markdown, TXT, HTML, DOCX, PDF, EPUB, ODT, RTF, DOC.
- Retrieval with vectorized numpy scoring and an optional `sqlite-vec`
  backend; opt-in cross-encoder reranking and semantic chunking.
- Interfaces: CLI, interactive shell, watch daemon, a read-only MCP server,
  and an opt-in local HTTP API + web UI (Starlette).
- Cognition modules: oracle Q&A, tagger, chronicler (note freshness),
  mirror (contradiction detection), claims, synthesizer, and graph export.
- Multi-vault profiles for fast context switching.
- GitHub Actions CI across Python 3.11 / 3.12 / 3.13, plus ruff linting.
- `[build-system]` declaration in `pyproject.toml` so editable installs no
  longer rely on a deprecated setuptools fallback.

### Fixed
- Preserve intentional exception suppression with `raise ... from None` at the
  CLI/validation boundaries that convert internal errors into clean exits.

[Unreleased]: https://github.com/kahz12/Grimore-MD/compare/v3.2.0...HEAD
[3.2.0]: https://github.com/kahz12/Grimore-MD/releases/tag/v3.2.0
[3.1.0]: https://github.com/kahz12/Grimore-MD/releases/tag/v3.1.0
