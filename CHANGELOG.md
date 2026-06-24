# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CHANGELOG.md` (this file).

### Removed
- `requirements.txt` — it was an unpinned duplicate of the dependencies in
  `pyproject.toml`, which is the single source of truth. Install with
  `pip install -e .` (add extras as needed, e.g. `pip install -e ".[serve]"`).

### Fixed
- Note and sidecar writes raised "a bytes-like object is required, not 'str'"
  on installs that resolved python-frontmatter 1.3.0, whose `dump()` no longer
  encodes when handed a binary file handle. `FrontmatterWriter` now serializes
  with `dumps()` and encodes explicitly, so writes work across frontmatter
  versions.
- Cleared all `ruff` findings (E402 import placement in `daemon.py` /
  `preflight.py`, explicit `strict=` on every `zip()`, an unused loop
  variable), and aligned the CI test job's dependencies with the suite.

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

[Unreleased]: https://github.com/kahz12/Grimore-MD/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/kahz12/Grimore-MD/releases/tag/v3.1.0
