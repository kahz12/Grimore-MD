"""Benchmark harness for the persistence and retrieval paths.

Measures the four paths the persistence and retrieval optimisations touch --
scan, ask, connect and dense-matrix load -- plus a direct count of SQLite
connection opens, which is the smoking gun for bottleneck B1.

Everything runs in-process against a stubbed LLM (see stub_llm.py for why).
In-process is not a convenience: counting `sqlite3.connect` calls requires
being inside the interpreter that makes them, and it also drops the ~0.3 s of
interpreter startup that a subprocess would fold into every timing, which is
noise of the same order as the +-5% reproducibility target.

Isolation: the harness builds its own grimore.toml in a work directory and
chdir's there, so it never reads the user's config, never touches the real
vault, and never writes to the repo's grimore.db.

Usage:
    python bench/measure.py --notes 2000
    python bench/measure.py --notes 200 --repeat 3      # variance check
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sqlite3
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import make_vault  # noqa: E402  (bench-local module, needs BENCH_DIR on the path first)
from stub_llm import StubServer  # noqa: E402

# Fixed question for the ask benchmark. Constant across runs by design: query
# text determines which chunks are retrieved and therefore how much work the
# context builder does, so varying it would vary the measurement.
ASK_QUESTION = "What are the retention policy thresholds for the cluster index?"

# grimore.toml for the isolated bench workspace. auto_commit is off and that is
# load-bearing, not tidiness: the bench vault lives inside the Grimore git repo,
# so an enabled GitGuard would snapshot thousands of generated notes into the
# project's own history on the first write.
_CONFIG_TEMPLATE = """\
[vault]
path = "{vault}"
ignored_dirs = [".obsidian", ".trash", ".git", "Templates"]

[cognition]
model_llm_local = "stub-llm"
model_embeddings_local = "stub-embed"
allow_remote = false
hybrid_search = true
rerank = false
vector_backend = "{vector_backend}"

[memory]
db_path = "{db}"

[output]
auto_commit = false
dry_run = false
"""


class ConnectionCounter:
    """Counts SQLite connections opened AND statements executed.

    Two distinct costs, and they stopped moving together once connection reuse
    landed. Connections are what per-thread reuse removes; statements are what
    an N+1 shows up in. Before the reuse, the connection count doubled as a
    query proxy because there was one connection per query; afterwards it reads
    ~1 for a whole scan and no longer discriminates, so the statement count is
    the metric that survives.

    The obvious alternative is `strace -e trace=openat` for the connection half.
    Counting at the sqlite3 layer instead is portable (strace needs ptrace
    permission, and is absent on plenty of hosts) and strictly more precise: it
    counts the connections Grimore actually opens rather than every file-open
    the process makes against the db path, including the -wal and -shm
    sidecars. It also has no way to see statements on an already-open
    connection, which is exactly the reused-connection case.
    """

    # Statement tracing is opt-in because it is not free: the callback fires
    # once per statement at ~1.2 us, and a single ask executes >17000 of them,
    # so leaving it on would add ~20 ms to a 100 ms measurement and report the
    # instrumentation as a regression. Timing runs and counting runs are
    # therefore separate passes -- the same reason tracemalloc is isolated in
    # bench_load_dense.
    trace_statements = False

    def __init__(self, db=None) -> None:
        self.count = 0
        self.queries = 0
        self.queries_internal = 0
        self._original = sqlite3.connect
        # Connections already open when the context is entered: a live
        # Session holds one for its lifetime, so a counter that only hooked
        # newly-opened connections would report zero statements for a query
        # that in fact did plenty of work.
        # Database._connections maps thread ident -> (Thread, Connection), so
        # take the values. Iterating the mapping directly would yield idents.
        registry = getattr(db, "_connections", None) or {}
        self._existing = (
            [conn for _thread, conn in registry.values()]
            if ConnectionCounter.trace_statements else []
        )

    def __enter__(self) -> "ConnectionCounter":
        def counting_connect(*args, **kwargs):
            self.count += 1
            conn = self._original(*args, **kwargs)
            if ConnectionCounter.trace_statements:
                conn.set_trace_callback(self._on_statement)
            return conn

        sqlite3.connect = counting_connect
        for conn in self._existing:
            conn.set_trace_callback(self._on_statement)
        return self

    def _on_statement(self, sql: str) -> None:
        # SQLite prefixes the statements a virtual table issues on its own
        # behalf with "--". FTS5's bm25() reads a docsize row per matching
        # document, so those dwarf everything else and would drown out the
        # application-level N+1 we care about. Counted, but separately.
        if sql.lstrip().startswith("--"):
            self.queries_internal += 1
        else:
            self.queries += 1

    def __exit__(self, *exc) -> None:
        sqlite3.connect = self._original
        for conn in self._existing:
            conn.set_trace_callback(None)


@contextlib.contextmanager
def quiet():
    """Swallow CLI output. Rich progress bars redraw thousands of times over a
    2000-note scan; the terminal writes are a real cost and would be measured
    as if they were Grimore's."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def run_cli(args: list[str]) -> None:
    """Invoke the Typer app in-process, without letting it exit the harness."""
    from grimore.cli import app
    with quiet():
        try:
            app(args, standalone_mode=False)
        except SystemExit:
            # Typer still raises SystemExit on some paths (e.g. --help style
            # short-circuits). A benchmark run should not die because of it.
            pass


def prepare_workspace(work: Path, vault: Path, notes: int, seed: int,
                      vector_backend: str) -> Path:
    """Regenerate vault + config + empty DB. Returns the config path.

    The vault is rebuilt every time because `scan` writes frontmatter back into
    the .md sources: a reused vault would arrive already tagged and fast-skip
    on the content hash, measuring the skip path instead of the ingest path.
    """
    work.mkdir(parents=True, exist_ok=True)
    make_vault.build(vault, notes, seed)

    db_path = work / "grimore.db"
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(db_path) + suffix)
        if target.exists():
            target.unlink()

    config_path = work / "grimore.toml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            vault=vault.as_posix(),
            db=db_path.as_posix(),
            vector_backend=vector_backend,
        ),
        encoding="utf-8",
    )
    return config_path


def bench_scan(work: Path) -> dict:
    """Cold scan of the freshly generated vault, with connection accounting."""
    counter = ConnectionCounter()
    start = time.perf_counter()
    with counter:
        run_cli(["scan", "--no-dry-run"])
    elapsed = time.perf_counter() - start
    return {
        "scan_s": elapsed,
        "scan_db_opens": counter.count,
        "scan_db_queries": counter.queries,
        "scan_db_queries_internal": counter.queries_internal,
    }


def bench_ask(session, warm_samples: int = 5) -> dict:
    """Ask timings, split into a cold and a warm measurement.

    The two model the two real callers, and conflating them was measuring
    neither. A one-shot `grimore ask` builds the dense matrix inside its only
    query, so its retrieve stage carries the full load_dense cost; the shell
    keeps a Session warm, so every query after the first hits the connector's
    signature-keyed cache. The cold number is therefore dominated by matrix
    construction and inherits its variance, while the warm number isolates the
    per-query work -- which is what the Oracle N+1 actually lives in, and what
    has to be readable if a drop in retrieve_s is to be provable.

    The Oracle already instruments embed/retrieve/rerank/generate separately
    (oracle.py:346-348), so these are its own numbers, not the harness's.
    """
    connector = session.oracle.connector

    connector._cache_sig = None
    cold_counter = ConnectionCounter(session.db)
    with cold_counter:
        cold = session.oracle.ask(ASK_QUESTION, top_k=5)
    cold_timings = dict(cold.get("timings") or {})

    warm_runs = []
    warm_counter = ConnectionCounter(session.db)
    with warm_counter:
        for _ in range(warm_samples):
            result = session.oracle.ask(ASK_QUESTION, top_k=5)
            warm_runs.append(dict(result.get("timings") or {}))

    warm: dict = {}
    for key in warm_runs[0]:
        values = [r[key] for r in warm_runs if isinstance(r.get(key), (int, float))]
        if values:
            warm[key] = statistics.median(values)
    # Per-query opens, not the total across samples: the metadata-query
    # budget is stated per ask, so the counter has to match that shape.
    warm["db_opens"] = warm_counter.count / warm_samples
    warm["db_queries"] = warm_counter.queries / warm_samples
    warm["db_queries_internal"] = warm_counter.queries_internal / warm_samples

    return {
        "ask_cold": {
            **cold_timings,
            "db_opens": cold_counter.count,
            "db_queries": cold_counter.queries,
            "db_queries_internal": cold_counter.queries_internal,
            "sources": len(cold.get("sources") or []),
        },
        "ask": warm,
    }


def bench_connect(work: Path) -> dict:
    """`connect` in dry-run: measures the O(notes x N) scan without the writes.

    Dry-run on purpose -- B4 is about the similarity sweep, and letting it
    inject links would mix in a vault-wide rewrite that no optimisation in the
    plan touches.
    """
    counter = ConnectionCounter()
    start = time.perf_counter()
    with counter:
        run_cli(["connect", "--dry-run"])
    return {
        "connect_s": time.perf_counter() - start,
        "connect_db_opens": counter.count,
        "connect_db_queries": counter.queries,
        "connect_db_queries_internal": counter.queries_internal,
    }


def bench_load_dense(session, samples: int = 5) -> dict:
    """Cold build of the dense scoring matrix: wall time and peak allocation.

    Timing and memory are measured in separate passes on purpose. tracemalloc
    hooks every allocation, and this path allocates heavily (a full fetchall
    plus the numpy matrix), so timing under the profiler measures the profiler
    as much as the code -- and its overhead is itself jittery, which was enough
    to blow past the +-5% reproducibility target on its own.

    The timed pass takes the median of several samples because this is a
    sub-100 ms operation on small vaults, where scheduler noise is the same
    order as the measurement.

    The connector caches on the embeddings signature, so the cache is cleared
    before every sample -- otherwise all but the first would time a dict lookup.
    Clearing that in-process key is not enough on its own: an earlier `ask` in
    the same run will have written the on-disk matrix, and every sample would
    then time a mmap hit rather than the build this function is named after. So
    the disk cache is dropped too, and the connector is pinned to the build
    path for the duration.
    """
    from grimore.utils import matrix_cache

    connector = session.oracle.connector
    was_enabled = connector.matrix_cache_enabled
    connector.matrix_cache_enabled = False
    matrix_cache.clear(session.db.db_path)

    durations = []
    for _ in range(samples):
        connector._cache_sig = None
        start = time.perf_counter()
        keys, matrix, _blobs = connector._load_dense()
        durations.append(time.perf_counter() - start)

    connector._cache_sig = None
    tracemalloc.start()
    connector._load_dense()
    current, peak_bytes = tracemalloc.get_traced_memory()
    peak = peak_bytes / 1e6
    # What the load leaves behind, as opposed to the transient high-water mark.
    # A one-shot CLI only cares about the peak; a shell Session or the daemon
    # holds the resident figure for its whole lifetime, so both are recorded.
    resident = current / 1e6
    tracemalloc.stop()

    connector.matrix_cache_enabled = was_enabled
    connector._cache_sig = None
    return {
        "load_dense_s": statistics.median(durations),
        "load_dense_peak_mb": peak,
        "load_dense_resident_mb": resident,
        "load_dense_rows": len(keys or []),
        "load_dense_matrix_mb": (matrix.nbytes / 1e6) if matrix is not None else 0.0,
    }


def run_once(work: Path, vault: Path, notes: int, seed: int,
             vector_backend: str) -> dict:
    prepare_workspace(work, vault, notes, seed, vector_backend)

    previous_cwd = Path.cwd()
    os.chdir(work)
    try:
        from grimore.session import Session
        from grimore.utils.config import load_config

        results: dict = {}
        results.update(bench_scan(work))

        # A fresh Session for the read-side benchmarks: the scan ran through
        # the CLI's own short-lived session, and reusing a warm handle here
        # would hide the cold-start cost that the CLI actually pays.
        session = Session(load_config())
        try:
            # Warm the embedder's HTTP connection before timing anything.
            # requests keeps a pooled connection per host, so the first call
            # pays TCP setup plus the stub's thread spawn -- tens of
            # milliseconds landing entirely inside ask's embed_s stage, which
            # is otherwise a few milliseconds. Left in, that one-off cost
            # dominated the spread and made ask timings fail the +-5% target.
            session.embedder.embed("benchmark warmup")

            # ask is measured before load_dense, not after. bench_load_dense
            # rebuilds the full row list and matrix several times over; the
            # resulting allocation churn and GC pressure landed on whatever ran
            # next, and ask's stages are single-digit milliseconds -- small
            # enough that the churn, not the code, decided the number.
            results.update(bench_ask(session))
            results.update(bench_load_dense(session))
        finally:
            session.close()

        results.update(bench_connect(work))
        return results
    finally:
        os.chdir(previous_cwd)


def summarize(runs: list[dict]) -> dict:
    """Aggregate repeated runs and report spread against the +-5% target."""
    summary: dict = {}
    scalar_keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    for key in scalar_keys:
        values = [r[key] for r in runs]
        mean = statistics.fmean(values)
        spread = (max(values) - min(values)) / mean * 100 if mean else 0.0
        summary[key] = {
            "mean": mean,
            "min": min(values),
            "max": max(values),
            "spread_pct": spread,
            "within_5pct": spread <= 5.0,
        }
    for section in ("ask", "ask_cold"):
        section_runs = [r[section] for r in runs if section in r]
        if not section_runs:
            continue
        summary[section] = {}
        for key in section_runs[0]:
            values = [a[key] for a in section_runs if isinstance(a.get(key), (int, float))]
            if not values:
                continue
            mean = statistics.fmean(values)
            spread = (max(values) - min(values)) / mean * 100 if mean else 0.0
            summary[section][key] = {
                "mean": mean,
                "min": min(values),
                "max": max(values),
                "spread_pct": spread,
                "within_5pct": spread <= 5.0,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--vector-backend", default="numpy",
                        help="Pin the backend so the baseline states which "
                             "path it measured; 'auto' would silently switch "
                             "if sqlite-vec becomes loadable.")
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "results.json")
    parser.add_argument("--work", type=Path, default=BENCH_DIR / "work")
    parser.add_argument("--vault", type=Path, default=BENCH_DIR / "vault")
    parser.add_argument("--count-queries", action="store_true",
                        help="Trace SQL statements. Adds ~1.2 us per statement, "
                             "so use it for counting runs, not timing runs.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the generated vault and DB after the run.")
    args = parser.parse_args()

    ConnectionCounter.trace_statements = args.count_queries

    with StubServer() as stub:
        # Both the router and the embedder read OLLAMA_HOST at construction
        # (llm_backends/ollama.py:37, embedder.py:46), so pointing the env var
        # at the stub is enough -- no config surgery, and SecurityGuard's
        # loopback validation still runs exactly as in production.
        os.environ["OLLAMA_HOST"] = stub.url

        runs = []
        for iteration in range(args.repeat):
            print(f"run {iteration + 1}/{args.repeat} (notes={args.notes})...",
                  file=sys.stderr, flush=True)
            runs.append(run_once(args.work, args.vault, args.notes, args.seed,
                                 args.vector_backend))

    payload = {
        "config": {
            "notes": args.notes,
            "seed": args.seed,
            "repeat": args.repeat,
            "vector_backend": args.vector_backend,
            "trace_statements": args.count_queries,
            "python": sys.version.split()[0],
            "llm": "stub (bench/stub_llm.py)",
        },
        "runs": runs,
        "summary": summarize(runs) if runs else {},
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)

    if not args.keep:
        shutil.rmtree(args.vault, ignore_errors=True)
        shutil.rmtree(args.work, ignore_errors=True)


if __name__ == "__main__":
    main()
