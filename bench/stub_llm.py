"""Deterministic Ollama-shaped stub server for the benchmark harness.

Why the benchmark must not talk to a real Ollama
------------------------------------------------
The Hito 0 definition of done demands a baseline reproducible to within +-5%
between runs. Live LLM generation cannot meet that: token throughput drifts
with model residency, thermal state and whatever else shares the GPU. Worse,
`scan` calls `tagger.tag_note()` once per changed note (cli.py:279, before the
dry-run check), so on a real backend LLM latency is ~90% of scan wall-clock --
which would bury the very thing the persistence optimisations target. A -30%
improvement in the SQLite layer is invisible when SQLite is 5% of the total.

So the stub answers instantly and deterministically, and the harness measures
the layer under optimisation: parsing, chunking, SQL and numpy. What this
deliberately does NOT measure is answer quality or real end-to-end latency;
those belong to `grimore eval` against real models, which is the separate gate
the plan already assigns to opt.6.

Embedding fidelity
------------------
Vectors come from a hashing vectorizer over the note's words rather than from
a hash of the whole string. A whole-string hash would make every vector
mutually orthogonal, so `connect` would find no candidate above its 0.7
threshold and the top-k paths would degenerate into measuring nothing. Bucketed
word counts preserve the property that matters here: notes sharing vocabulary
score higher, so ranking does real work. The vectors carry no semantics beyond
lexical overlap, which is why quality claims are out of scope for this harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Matches the dimensionality of the embedding models this vault is configured
# for. The absolute value is arbitrary for the benchmark, but it must stay
# fixed across a baseline pair: changing it changes every stored vector, the
# matrix size and therefore every number the harness reports.
DIM = 768

_WORD_RE = re.compile(r"[a-z0-9]+")

# Canned tagger reply. The router parses this into a dict and the tagger
# reconciles the tags against the vault taxonomy, so the strings exercise the
# real reconciliation path rather than short-circuiting it.
_TAG_POOL = ["infra", "retention", "vector", "ops", "theory", "index", "research"]


def embed(text: str) -> list[float]:
    """Hashing vectorizer: bucket word counts into DIM dims, then L2-normalize.

    Deterministic across runs and processes -- md5 of the word is stable where
    Python's own hash() is salted per interpreter, which would silently break
    reproducibility between the "before" and "after" runs of a baseline.
    """
    vec = [0.0] * DIM
    for word in _WORD_RE.findall(text.lower()):
        digest = hashlib.md5(word.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % DIM
        # Sign bit from a second slice so buckets can cancel, which keeps
        # unrelated documents from all drifting toward the same direction.
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # Degenerate input (no word characters). Return a fixed unit vector
        # rather than zeros: a zero vector would make cosine undefined and the
        # normalizer downstream divide by zero.
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


def tag_response(text: str) -> str:
    """A stable tags/summary/category JSON string, keyed off the content.

    Derived from the text so different notes get different tags (the taxonomy
    reconciler and the note_tags junction then do real work), but identical
    for identical input so a re-run reproduces the previous vault state.
    """
    digest = hashlib.md5(text.encode("utf-8")).digest()
    tags = sorted({_TAG_POOL[digest[i] % len(_TAG_POOL)] for i in range(3)})
    return json.dumps({
        "tags": tags,
        "summary": f"Synthetic summary {digest[:4].hex()} for benchmark input.",
        "category": _TAG_POOL[digest[5] % len(_TAG_POOL)],
    })


class _Handler(BaseHTTPRequestHandler):
    # Silence per-request logging: at several thousand requests per scan the
    # stderr writes are themselves a measurable cost and would pollute timings.
    def log_message(self, *args) -> None:
        return

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # Preflight and the router's health check hit /api/tags.
        if self.path.startswith("/api/tags"):
            self._json({"models": [{"name": "stub-llm"}, {"name": "stub-embed"}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return

        if self.path.startswith("/api/generate"):
            self._json({"response": tag_response(payload.get("prompt", "")), "done": True})
        elif self.path.startswith("/api/embeddings"):
            self._json({"embedding": embed(payload.get("prompt", ""))})
        elif self.path.startswith("/api/embed"):
            texts = payload.get("input") or []
            if isinstance(texts, str):
                texts = [texts]
            self._json({"embeddings": [embed(t) for t in texts]})
        else:
            self.send_error(404)


class StubServer:
    """Context manager wrapping the stub on an ephemeral loopback port.

    Binds to 127.0.0.1 explicitly rather than localhost: SecurityGuard pins the
    resolved loopback address, and on hosts where localhost resolves to ::1
    first the pin and the bind would disagree.
    """

    def __init__(self, port: int = 0) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "StubServer":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11500)
    args = parser.parse_args()
    with StubServer(args.port) as server:
        print(f"stub LLM on {server.url} (OLLAMA_HOST={server.url})", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
