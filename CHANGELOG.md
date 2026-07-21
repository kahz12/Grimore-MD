# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
