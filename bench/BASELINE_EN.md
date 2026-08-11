# Grimore — Performance baseline

*[Versión en español](BASELINE_ES.md)*

> Starting numbers the performance work is compared against. Each optimisation
> below adds an "after" column to these tables.
>
> On cross-references: optimisations are numbered (opt. 1–10) and bottlenecks are named (B1–B4)
> after internal design documents that are **not part of this repository**. The labels are kept so
> the log in §8 stays traceable against the history; every entry explains on its own what changed,
> why, and with which numbers, without needing them.

**Measured at**: `2ef8afa` (v3.2.0 + bug fix) · 2026-08-03

---

## 1. Reproducing it

```bash
python bench/measure.py --notes 2000 --repeat 2      # the baseline published here
python bench/measure.py --notes 200  --repeat 3      # quick variance check
```

The harness is self-contained: it generates its own vault, writes its own `grimore.toml` into a
throwaway work directory and `chdir`s there. It never reads your config, never touches your vault,
and never writes to the repo's `grimore.db`. Generated artifacts are gitignored; `make_vault.py`
reproduces the vault byte-for-byte from its seed.

| File | What it is |
|---|---|
| `bench/make_vault.py` | Deterministic vault generator (fixed seed) |
| `bench/stub_llm.py` | Deterministic server shaped like the Ollama API |
| `bench/measure.py` | Measurement harness → `bench/results.json` |

---

## 2. Environment

| | |
|---|---|
| CPU | Intel Core i7-6500U @ 2.50 GHz (4 threads) |
| RAM | 15 GB |
| Disk | SSD, ext4 |
| Kernel | 7.0.0-28-generic |
| Python | 3.14.4 (`venv/`) |
| Vector backend | **numpy** (pinned with `--vector-backend numpy`) |
| LLM | deterministic stub (`bench/stub_llm.py`) |

> **Correction about sqlite-vec.** Sections §4–§8 were measured **without** sqlite-vec, and claimed
> the extension "does not load on this Python 3.14 build". That was false: the wheel simply wasn't
> installed, and `_probe_vec_extension` returns `False` on an `ImportError` exactly as it does on a
> load failure, so both cases look identical from outside. Once installed (`pip install
> sqlite-vec`) it loads fine (v0.1.9) and the 5 tests marked `vec` pass.
>
> Consequence for the measurements: with the extension present, every embedding **dual-writes** to
> `embeddings_vec`, so a scan does more work than the §4 baseline. That is one more reason timings
> are only meaningful A/B within a single session (§9). The baseline pins `--vector-backend numpy`
> explicitly so ranking always takes the same path; that does not disable the mirror, which lives in
> the write layer.

---

## 3. Why the LLM is stubbed (a methodology decision)

It is tempting to measure `scan` against a real Ollama. Reading the code, that doesn't work, for
two reasons:

1. **`tagger.tag_note()` runs at `cli.py:279`, before the `dry_run` check.** A scan pays one LLM
   call per note even in dry-run.
2. **With the LLM in the loop, generation is ~90% of wall-clock.** The acceptance criterion for
   opt. 1 was **−30% or more on scan**, but a 30% improvement in the SQLite layer is undetectable
   when SQLite is 5% of the total. Generation latency also drifts with model residency and thermal
   state, which makes the ±5% reproducibility target impossible.

The stub answers instantly and deterministically, so the harness measures **the layer opts. 1–5
actually touch**: parsing, chunking, SQL and numpy.

**What this harness deliberately does NOT measure**: answer quality, retrieval quality, and real
end-to-end latency. That is `grimore eval`'s job against real models. The stub's vectors come from
a hashing vectorizer over words, so they preserve lexical similarity (ranking does real work and
`connect` finds candidates) but they **do not encode semantics**. No quality claim can rest on
these numbers.

---

## 4. Baseline — 2000 notes, seed 42, 2 runs

Generated vault: 2000 notes · 17,465 chunks · ~9.4 KB per note.

### 4.1 Reference metrics

| Metric | Mean | Min | Max | Spread | ±5% | Bottleneck / opt. |
|---|---:|---:|---:|---:|:---:|---|
| `scan_s` | **476.09 s** | 474.94 | 477.25 | 0.48% | yes | B1 · opt. 1, 3 |
| `scan_db_opens` | **66,396** | 66,396 | 66,396 | 0.00% | yes | **B1 · opt. 1** |
| `connect_s` | **26.46 s** | 26.37 | 26.54 | 0.63% | yes | B4 · opt. 5 |
| `connect_db_opens` | **12,003** | 12,003 | 12,003 | 0.00% | yes | B4 · opt. 5 |
| `load_dense_s` | **0.1122 s** | 0.1103 | 0.1141 | 3.39% | yes | B3 · opt. 4 |
| `load_dense_peak_mb` | **121.22 MB** | 121.22 | 121.22 | 0.00% | yes | **B3 · opt. 4** |
| `load_dense_rows` | 17,465 | — | — | 0.00% | yes | context |
| `load_dense_matrix_mb` | 53.65 MB | — | — | 0.00% | yes | context |

### 4.2 `ask` with `top_k=5` — warm and cold

Measured separately because they are the **two real callers**. A one-shot `grimore ask` builds the
dense matrix inside its only query, so its `retrieve_s` carries the full `load_dense` cost and
inherits its variance. The shell keeps a `Session` alive, so every query after the first hits the
connector's signature-sealed cache. The Oracle's N+1 (B2, opt. 2) lives in the warm path; measuring
them together measured neither.

| Metric | Mean | Spread | ±5% | Note |
|---|---:|---:|:---:|---|
| `warm.total_s` | **0.1010 s** | 0.68% | yes | shell / live `Session` |
| `warm.retrieve_s` | **0.0894 s** | 0.35% | yes | **opt. 2 gate** |
| `warm.embed_s` | 0.0009 s | 1.03% | yes | embedding cache |
| `warm.db_opens` | **13.0** | 0.00% | yes | **opt. 2 gate** (target: ≤2) |
| `warm.generate_s` | 0.0026 s | 12.81% | no | 2.6 ms of stub; no opt. touches it |
| `cold.total_s` | 0.2289 s | 16.59% | no | one-shot CLI |
| `cold.retrieve_s` | 0.2057 s | 18.06% | no | dominated by matrix build |
| `cold.db_opens` | **15.0** | 0.00% | yes | +2 over warm: matrix load |

**About the four metrics that miss ±5%**: none is a gate. `warm.generate_s` and `cold.rewrite_s`
are millisecond or zero-cost stages (with no history, `_rewrite_query` returns immediately) that no
optimisation attacks. `cold.total_s` / `cold.retrieve_s` are dominated by `load_dense`, whose cost
is already measured in isolation and stably as `load_dense_s`. The acceptance criteria for opts. 1,
2 and 4 are stated in **counts** ("O(chunks) → O(1)", "~11 queries → ≤2"), and all four counters
have **exactly 0.00% spread**.

### 4.3 Variance check at 200 notes (n=3)

The published baseline uses n=2 because of wall-clock cost (~8 min per run). ±5% reproducibility
was demonstrated independently with n=3 at 200 notes: `scan_s` 4.83%, `connect_s` 4.67%,
`warm.retrieve_s` 0.59%, `warm.total_s` 4.09%, and every counter at 0.00%. The one miss at that
scale is `load_dense_s` (30.8%), which at 200 notes is a 15 ms measurement where scheduler jitter
dominates; at baseline scale (2000 notes, 112 ms) it drops to 3.39%.

---

## 5. What the baseline already proves

### B1 — a connection per operation (critical, opt. 1)

**66,396 SQLite connection opens to index 2000 notes / 17,465 chunks** = **3.80 opens per chunk**.
Each one re-applies `PRAGMA journal_mode=WAL` and does its own `commit`.

The damning figure: `476.09 s / 66,396 = 7.17 ms per connection`. With an LLM that answers
instantly, the scan is **dominated by the WAL `fsync` on every commit**. That is what makes opts. 1
and 3 the highest-leverage block of the whole effort.

`Session.close()` documents it explicitly at `session.py:220`:

> *"Database holds no long-lived connection — each call opens one and closes it on exit"*

### B2 — the Oracle's N+1 (high, opt. 2)

**13 connections per warm query** (15 cold) for an `ask` with `top_k=5`. The estimate was ~11; the
real number is slightly worse. Target: ≤2.

### B3 — `_load_dense` drags `text_content` along (high, opt. 4)

**121.22 MB peak for a 53.65 MB matrix** = **67.57 MB (2.26×) of overhead**. That delta is exactly
the `text_content` that `get_all_embeddings_with_id` fetches and the matrix does not need — the
hypothesis, now measured. And this with only 17,465 chunks; a large vault projects to 100k.

### B4 — `connect` is O(notes × N) (high, opt. 5)

**26.46 s and 12,003 connection opens** for 2000 notes = **6.0 opens per note**, on top of the full
per-note similarity sweep. Confirms the B1+B2+B4 pile-up.

---

## 6. On counting connections

The obvious approach is `strace -f -e trace=openat`. The harness instead counts `sqlite3.connect`
calls inside the process, because that is portable (`strace` needs ptrace permission and isn't
everywhere) and more precise. **Both methods were cross-validated over an identical cold scan of 50
notes**:

| Method | Count |
|---|---:|
| In-process `sqlite3.connect` counter | **1,725** |
| `strace` · `openat` on `grimore.db` | **1,725** |
| `strace` · `openat` on `grimore.db-wal` | 1,725 |
| `strace` · `openat` on `grimore.db-shm` | 1,725 |
| `strace` · total syscalls | 5,175 |

Exact agreement on the main file, and it shows why counting at the sqlite3 layer is cleaner:
**each connection opens three files** (main + WAL + shm), so the raw `openat` total triples the
connection count.

---

## 7. Baseline definition of done

- [x] `bench/results.json` is produced by one command and is reproducible within ±5% on every gate
      metric (counters exactly at 0.00%).
- [x] Baseline recorded for the 5 metrics: `scan_s`, per-stage `ask` timings, `connect_s`,
      `load_dense_s` + `peak_mb`, and the connection-open count.
- [x] Full suite green before starting: **907 passed, 11 skipped, 0 failed**; `ruff check grimore`
      clean.

The 11 skips are missing dependencies, not failures: 5 `e2e` (the `nomic-embed-text` model isn't
pulled; the vault uses `nomic-embed-text-v2-moe`), 5 `vec` (sqlite-vec not installed at the time)
and 1 `reranker` (no sentence-transformers).

---

## 8. Optimisation log

### opt. 8 — Magic constants → configuration

Six constants promoted to `grimore.toml` keys, all read with `getattr(..., default)` following the
existing house pattern:

| Key | Section | Default | Constant it replaces |
|---|---|---:|---|
| `chunk_store_chars` | `[cognition]` | 500 | `reembed.py` `text_truncation` |
| `context_max_chars` | `[cognition]` | 16,000 | `oracle.py` `_ORACLE_CONTEXT_MAX_CHARS` |
| `embed_batch_size` | `[cognition]` | 32 | `embedder.py` `_EMBED_BATCH_SIZE` |
| `circuit_failure_threshold` | `[cognition]` | 5 | `llm_router.py` `_FAILURE_THRESHOLD` |
| `circuit_cooldown_s` | `[cognition]` | 120 | `llm_router.py` `_COOLDOWN_SECONDS` |
| `max_turns` | `[shell]` | 3 | `session.py` `MAX_TURNS` |

**No performance change, and that is the check.** Being pure configuration exposure, the acceptance
criterion is that the defaults reproduce the previous behaviour. Verified by running the harness at
200 notes after the change: **every deterministic counter identical to baseline** (`scan_db_opens`
6,564, `connect_db_opens` 1,200, `load_dense_rows` 1,721, `load_dense_matrix_mb` 5.2869,
`load_dense_peak_mb` 11.7633, `warm.db_opens` 13, `cold.db_opens` 15). That is equivalence shown
empirically, not merely asserted by a test.

Implementation details worth recording:

- **Class-level default pattern.** `Oracle`, `Session` and `LLMRouter` declare the default as a
  class attribute and shadow it with the configured value in `__init__`. Necessary because several
  tests build these objects via `__new__` to isolate them from the DB and the LLM; without the
  class attribute, production code would blow up with `AttributeError` on that path.
- **`max_turns = 0` really disables memory.** The cut is checked before the slice: `turns[-0:]` is
  `turns[0:]`, so folding zero into the slice would have kept the whole history instead of clearing
  it. Covered by a test.
- **`embed_batch_size` is floored at 1**, because a zero step would make `embed_batch`'s `range()`
  loop never advance.

Tests: `tests/test_config.py::TestTunableDefaults` (load and override) and `tests/test_tunables.py`
(15 cases asserting the values **reach** their consumer). The distinction matters: load tests alone
would pass even if a consumer were still reading its module constant. All five wirings were
validated red→green by reverting them one at a time — an exercise that also exposed a gap: the
first version of the embedder test only checked `__init__.batch_size` and did not fail when the
consumption inside `embed_batch` was reverted.

Suite after the change: **925 passed, 11 skipped, 0 failed** · `ruff` clean.

---

### opt. 1 — Reusable per-thread SQLite connection

One connection per thread, alive for the lifetime of the `Database`, with the PRAGMAs and the
sqlite-vec load applied **once** instead of on each of the 73 data-access paths. Adds
`busy_timeout=5000`, `cache_size=-16000` and `mmap_size=256MB` (useless before: a cache discarded
microseconds later never hits). Explicit teardown via `Database.close()`, called from
`Session.close()` and `daemon.stop()`.

| Metric | pre-opt.1 | opt.1 | Change | Criterion |
|---|---:|---:|---:|---|
| `scan_db_opens` | 66,396 | **1** | −100% | O(1) per thread, met |
| `connect_db_opens` | 12,003 | **1** | −100% | met |
| `scan_s` | 529.75 s | **65.14 s** | **−87.7%** | −30% or more, met |
| `connect_s` | 50.89 s | **31.72 s** | **−37.7%** | (not required) |

Measured A/B in one session, 2000 notes, 2 runs per variant. See §9 for why timings are measured
this way and not against the §4 table.

**Design.** Three pieces worth recording:

- **Connection registry + generation counter.** `threading.local` gives no way for one thread to
  clear another's slot, but the daemon indexes on the watcher thread while the main thread reads,
  so `close()` has to reach both. The registry gathers them; the generation counter invalidates
  cached handles without touching foreign storage, and each thread notices and reopens on its next
  call. That is what makes reusing a `Database` after `close()` work.
- **`check_same_thread=False` is safe here** only because `_thread_conn` guarantees exactly one
  connection per thread. The flag exists so `close()` can reclaim foreign connections at shutdown,
  not so they can be shared.
- **VACUUM is unaffected**: `upkeep.py:30` already opened its own dedicated connection with
  `isolation_level=None`, because VACUUM refuses to run inside a transaction.

**The verification that had to come first.** The real risk of sharing a connection is not
performance but **re-entrancy**: if a method opens a connection and calls another that opens one
too, the inner `with conn:` would commit the outer transaction early, turning a rollback-on-error
into a partial write. Checked two independent ways before touching anything — a runtime probe
across all 925 tests plus the harness, and a static AST analysis — finding **0 nesting sites across
73 connection-opening methods**. `TestNoReentrancy` now guards the invariant.

**Functional verification against real Ollama** (not the stub), on a test vault:

| Surface | Result |
|---|---|
| `preflight`, `scan`, `status`, `ask`, `connect`, `tags`, `dedupe` | OK; `ask` answers with the right citation |
| Idempotence | Second scan: 0 processed, 2 unchanged (fast-skip via double hash) |
| Frontmatter | Written correctly into the source note |
| HTTP API | 200 requests: descriptors 8 → 12 and stable; no errors |
| Daemon | Indexes and reindexes; descriptors 11 → 15 → 15; clean SIGTERM stop |
| MCP | Handshake, `tools/list` and `tools/call` correct |
| **Concurrent daemon + CLI** | 50 CLI reads with the daemon indexing: **0 errors**, `integrity_check` ok, 0 orphan embeddings |

**Concurrency gate (non-negotiable).** `tests/test_connection_reuse.py`, 16 cases: one connection
per thread and reused, distinct connections across threads, `close()` actually releasing (plus a
descriptor count from `/proc/self/fd`, the inverse regression of the v3.2.0 fix), reuse after close,
idempotence, dead-thread reaping that leaves live ones alone, and the gate proper — one writer with
three readers and three concurrent writers with no `database is locked`, no lost rows, and
`PRAGMA integrity_check` returning `ok`. Validated red→green: with `busy_timeout=0` the concurrent
write test fails; with 5 s it passes.

**Descriptor leak on short-lived threads (found and fixed during verification).** The first version
stored connections in a list. Thread-local storage dies with the thread, but the list held a strong
reference, so the connection and its descriptor survived until `close()`. Reproduced: 100
short-lived threads left **101 connections and 207 open descriptors**, growing linearly; with the
usual 1024-FD limit, a long-lived daemon or API would end at `EMFILE`. Reachable in production via
`ThreadPoolExecutor` — which is anyio's threadpool model, the one Starlette runs sync handlers on.

The fix: the registry becomes a dict keyed by thread, and opening a new connection first sweeps and
closes those belonging to dead threads. Sweeping there rather than on every call keeps it off the
hot path: that branch runs once per thread, not once per query. Liveness is checked against the
`Thread` object rather than the ident, because the OS recycles idents and a new thread could
inherit a dead one's. After the fix connections stay bounded (2–9 in the same tests) and descriptors
flat.

Suite after the change: **941 passed, 11 skipped, 0 failed** · `ruff` clean · `mypy` no issues.

---

### opt. 3 — Batched embedding inserts

`store_embeddings_bulk` with `executemany` and **one transaction per note** instead of one per
chunk. `reembed_note` accumulates the rows and calls once; the singular `store_embedding` is kept
untouched as the rollback path and because a couple dozen tests use it.

Measured A/B in one session, 600 notes, with sqlite-vec installed (dual-write active):

| Metric | per chunk | batched | Change |
|---|---:|---:|---:|
| `scan_s` | 16.234 s | **13.610 s** | **−16.2%** |
| `scan_db_queries` | 65,375 | **56,839** | **−13.1%** |

On top of the −87.7% opt. 1 already delivered. The real saving is the `fsync`: in WAL every commit
pays one, and the old loop committed once per chunk. `executemany` merely removes the per-row Python
overhead on top of that.

**Why rowids are read back instead of computed.** The obvious shortcut is
`first_id = cur.lastrowid - len(rows) + 1`. That code **does not work**: since Python 3.11
`cursor.lastrowid` is `None` after an `executemany`, so the subtraction raises `TypeError`.
Verified, and the parity tests reject it (3 failures when substituted). A `MAX(id)` variant *would*
be correct inside the transaction — WAL allows a single writer, so nobody can interleave a row while
the lock is held — but the read-back doesn't need that reasoning to be correct and doesn't break if
the locking model changes later.

The read-back selects by `note_id` rather than an `IN (chunk_index, ...)`: a large PDF can exceed
SQLite's host-parameter limit, and the extra rows (kept chunks, not re-embedded ones) are simply
never consulted.

**Vec mirror parity gate.** `tests/test_bulk_embeddings.py`, 14 cases, 6 of which exercise
`embeddings_vec` with the real extension: equal row counts, **each vector filed under its own
rowid** (the failure a shifted mapping produces is a citation pointing at the wrong note — silent,
and only detectable by comparing the vector against its source row), survival of an incremental
re-embed, shrinking when the note shortens, no cross-note contamination, and a dimension mismatch
that still persists the source row. Plus a test that ten chunks produce **exactly one COMMIT**.

**Pre-existing bug fixed during verification: the vec mirror could break silently.**
`_create_vec_table` sets `self._vec_dim` in Python *inside* the transaction that creates
`embeddings_vec`. If that transaction later rolls back, the DDL is undone but the attribute
survives, so every later write skips creation and inserts into a table that no longer exists. The
`except OperationalError` downgraded it to a `warning` and the scan continued: `embeddings` kept
saving and **no vector was mirrored**, leaving a vec index that answers with a fraction of the
vault. Pre-existing, not introduced by opt. 3 — reproduced identically against `store_embedding`,
the single-row path this work does not touch.

The fix recovers **within the same call** rather than on the next one: the failing batch's
`embeddings` rows commit either way, so nobody would ever come back to mirror them. On a failed
write it clears `_vec_dim`, recreates the table (`CREATE ... IF NOT EXISTS`, so the retry is safe
even if the failure was something else) and retries once. This follows the convention
`drop_vec_table` already used, resetting the dim after an explicit DROP; the rollback path was
simply missing the equivalent. Covered by 3 tests, validated red→green.

Suite after the change: **963 passed, 6 skipped, 0 failed** · `ruff` clean · `mypy` no issues.
Without sqlite-vec installed (CI's default configuration): 949 passed, 20 skipped, 0 failed.

---

### opt. 2 — Killing the Oracle's N+1

The minimal proposal, extended to the anchors as well: `get_note_titles(ids)` and
`get_chunk_anchors_bulk(pairs)`. Retrieval is untouched — no `JOIN`, no change to the match
semantics — so pushing the metadata into the retrieval query stays available if a future bench
justifies it.

**Why `embedding_id` wasn't used.** Both retrieval paths already know it and drop it
(`connector.py`, "Drop the internal embedding_id"), and a test pins that contract. With it, anchors
would be a primary-key lookup instead of a `text_content` match. That is the faster and strictly
more correct route — two chunks of a note with the same stored 500 characters can today return each
other's anchor — but it changes observable semantics and breaks a tested contract. It is the natural
next step.

#### Gate 1 — query budget (session-independent)

200-note vault, `--count-queries`, A/B in one session:

| Metric | before | after | Change |
|---|---:|---:|---:|
| `ask` (warm) application statements | 13 | **5** | **−8** |
| └ of which metadata | 10 | **2** | **−80%** |
| `ask_cold` application statements | 17 | **9** | **−8** |

10 → 2 is exactly the acceptance criterion ("~11 → ≤2"). The 3 remaining warm statements are
transaction control, not metadata.

#### Gate 2 — citation parity

`_build_context` over the same vault and the same question, before and after: **`context` identical
byte for byte** (2814 chars) and `retrieved` identical (same note_ids, same ranks, same scores).

A difference did show up in `sources` during this check: **same set, different order**. It is not an
opt. 2 regression — `_build_context` returns `list(set(sources))` (`oracle.py:472`), and the
iteration order of a set of strings depends on `PYTHONHASHSEED`, which is randomised per process.
Verified by running the **same** code in three processes: three different orders, identical context.
Pre-existing nondeterminism, documented in the code itself ("``sources`` is flattened through
``set()`` and can't carry order"). Recorded as a wart, not a blocker.

#### Gate 3 — timing

15 runs per arm, 5 alternating `before/after` rounds to average out thermal drift (§9), 200 notes:

| `ask` warm | before (median) | after (median) | Change | IQRs |
|---|---:|---:|---:|---|
| uninstrumented gap | 2.251 ms | **1.340 ms** | **−40.5%** | barely overlap |
| `retrieve_s` | 11.194 ms | 11.336 ms | +1.3% | — |
| `embed_s` (control) | 0.119 ms | 0.119 ms | +0.2% | — |
| `generate_s` | 3.556 ms | 2.967 ms | −16.6% | **overlap** |
| `total_s` | 17.946 ms | 15.892 ms | −11.4% | — |

**The stated criterion pointed at the wrong bucket.** It asked for `retrieve_s` to drop, but the
Oracle stops that stopwatch (`oracle.py:380`) *before* the metadata loop. The work opt. 2 removes
falls in the gap between `retrieve_s`/`rerank_s` and `generate_s`, which no instrumented stage
covers — hence the "gap" row (`total_s` minus the sum of the stages). There the effect is
unambiguous: −40.5%, ~0.9 ms per `ask`.

`generate_s` dropping 16.6% is **not attributable to this change**: the stub is deterministic and
the context is byte-identical, so the payload is the same. Across 15 samples the IQRs overlap
comfortably — it is noise. `embed_s`, serving as a control, stays flat at +0.2%, which confirms the
harness did not drift during the batch.

The absolute saving (~0.9 ms) is small because the vault is small and the DB is warm; it scales with
the retrieved pool size and weighs far more on slow storage.

Suite after the change: **981 passed, 6 skipped, 0 failed** · `ruff` clean · `mypy` no issues across
73 files. `tests/test_batch_metadata.py` adds 13 cases, with the three key mutations (`ORDER BY id`
reversed, pair filter removed, dedup removed) verified red→green.

### opt. 4 — Lighter `_load_dense` + on-disk matrix cache

Two halves. **A**: build the matrix from a vector-only load instead of one that
also drags `text_content`. **B**: persist the built matrix as a `.npy` beside the database and
reload it memory-mapped.

#### A — dropping `text_content` from the scoring pass

| Metric | before | opt. 4A | Change |
|---|---:|---:|---:|
| `load_dense_peak_mb` | 121.2 MB | **56.29 MB** | **−53.6%** |
| `load_dense_resident_mb` | 119.7 MB | **56.28 MB** | **−53.0%** |
| overhead above the matrix itself | 67.57 MB | **2.63 MB** | **−96%** |
| `ask_cold.retrieve_s` | 183.96 ms | 141.50 ms | −23.1% |
| `load_dense_s` | 91.5 ms | 76.2 ms | −16.7% |

**The first attempt only bought −6.9%.** `tracemalloc` reports the transient high-water mark, and
the naive version still held two full copies of the vector data at once: the row list from
`fetchall`, then the joined buffer. Streaming the rows into one growing `bytearray` — so each row's
bytes are released as they are appended — is what took the peak from 113 MB to 56 MB. The metric
`load_dense_resident_mb` was added at the same time: the peak is what a one-shot CLI pays, but the
resident figure is what a shell `Session` or the daemon holds for its whole life, and only the
second was actually improved by removing the text.

**Ragged vectors need an explicit check, not an inferred one.** With the vectors concatenated into
one buffer, a mixed-width table has no recoverable row boundaries, so it must be refused rather
than reshaped. Inferring raggedness from the buffer length is not sound: widths `[8, 4, 12]` sum to
exactly `3 × 8`, so a length-only test accepts data it has to reject. The uniformity test runs per
row, and the fallback fetches the per-row vectors separately, so the common path never pays for
that list.

**A regression found and fixed during the change.** With text no longer riding along,
`find_similar_notes` fetches it per call — and `connect` calls it once per note, which showed up as
`connect_s` **+15.5%**. Neither `connect` nor the graph's suggested edges read `hit["text"]`; they
use `note_id` and `score`. With `with_text=False` on those two paths, `connect_db_queries` is
**1233 → 1233, delta 0**. The +4.8% still visible on the clock afterwards was session drift, which
the counter settles.

The accepted cost is `ask.db_queries` **5 → 6**: one lookup per ask in exchange for not carrying
500 chars per chunk through the scoring pass.

#### B — the `.npy` cache

Fresh processes (what a one-shot CLI actually does), 2000 notes, median of 5:

| Scenario | Load | vs no cache |
|---|---:|---:|
| No cache (rebuild every time) | 72.00 ms | — |
| First run with cache (build + write 53 MB) | 102.00 ms | +42% |
| Cache hit (`mmap`) | **33.00 ms** | **−54%** |

Break-even is the second invocation: the first pays 30 ms extra, each one after saves 39 ms.

**The seal the design called for is unsafe, and this is demonstrable.**
`swap_embedding_migration` installs the re-embedded table with
`INSERT INTO embeddings (id, ...) SELECT id, ... FROM embeddings_migration`, preserving every id.
After switching embedding model the count and the max id are **identical** while every vector has
changed. For the Connector's in-process cache that is a one-process hazard; for a cache that
survives restarts it would serve the previous model's vectors indefinitely. Three layers instead:

1. The seal carries `(count, max_id, total_vector_bytes)` — no change of dimension survives it.
2. The migration swap calls `matrix_cache.clear()` — covers a same-dimension model swap, which no
   seal can see.
3. `load()` re-checks the matrix shape — covers a truncated `.npy` under a still-valid seal, which
   a crash between the two writes can leave behind.

**A correction to the design's memory claim.** It states that with `mmap_mode="r"` RSS stops
scaling with vault size. Measured, that does not hold for this access pattern: `matrix @ q` touches
every row, so the mapping is paged in fully and RSS is identical (+56.3 MB) with and without the
cache. What the mapping does buy is that those 51.2 MB are **file-backed** (confirmed in
`/proc/self/smaps`) rather than anonymous heap, so the kernel can drop and re-read them under
pressure instead of having to swap. A real benefit, but a different one, and invisible to RSS.

**Also corrected: a comment of mine that overclaimed.** The optimistic seal re-check before writing
the cache was documented as preventing a stale hit. It does not — in that scenario the signature is
identical and it is the shape check that saves you. All it actually avoids is writing a file that is
guaranteed to miss. The mutation run is what exposed this: removing the check failed no test, which
was the signal that the comment described something untested and untrue.

Not hooked into `maintenance run`, deliberately. `id` is an `INTEGER PRIMARY KEY`, so VACUUM
preserves it and the seal still matches afterwards; clearing there would force a needless rebuild.
Verified empirically rather than assumed.

Suite after the change: **1027 passed, 6 skipped, 0 failed** · `ruff` clean · `mypy` no issues
across 74 files. `tests/test_dense_loading.py` (22 cases) and `tests/test_matrix_cache.py`
(24 cases), with seven mutations verified red→green.

### opt. 5 — Vectorised `connect`

`connect` called `find_similar_notes` once per note, and each of those multiplied the query against
every chunk in the vault: O(notes × chunks) products issued one at a time. The sweep is now one
blocked `Q @ C.T`.

| Metric | per-note loop | blocked sweep | Change |
|---|---:|---:|---:|
| `connect_s` | 20.47 s | **3.116 s** | **−84.8%** |
| `connect_db_queries` | 6,040 | **4,041** | **−1,999** |
| `scan_s` (control) | 46.42 s | 46.84 s | +0.9% |

A/B alternated within one session, 2000 notes. The A/B isolates the driver: the batch method and
the first-chunk query are present in both arms, and only the loop in `cli.py` differs, so nothing
else can account for the delta. The query saving is one per note — `find_similar_notes` reads the
embeddings signature on every call, which the sweep does once.

**The design's note-level mean would have changed the results.** It proposes a matrix of per-note
vectors, "the mean of its chunks, or the first chunk". Those are not interchangeable here: the old
loop walked the chunk table and kept the first row it met for each note, so the query vector was
always the first chunk. `get_first_chunk_vectors` selects exactly that, now explicitly rather than
by luck of SQLite's scan order.

**Bit-exact parity is not achievable, and asserting it was wrong.** The per-query path is a
matrix-by-vector product (gemv); the batch is matrix-by-matrix (gemm). BLAS accumulates them in a
different order, so the scores differ in the last bits. Measured over 200 queries against 1000
chunks at 768 dims: largest disagreement **4.8e-07** (four float32 eps), median 7.5e-09, and
**0 of 200 queries had a different top-20**. Block size perturbs them too, so `block_rows` is not a
pure performance knob. The gate is therefore the ranking — same notes, same order — with scores
compared at a tolerance 20× the measured worst case.

**Blocked from the start, as required** — but the first version's blocking did nothing, and only a
review caught it. Iterating a numpy block yields row *views* whose base is the whole block, so
keeping one row per query to read its scores later pinned every block until the sweep ended.
Measured on 400 notes × 2000 chunks: `block_rows=16` peaked at 5.04 MB and `block_rows=400` at
4.98 MB — identical, when the per-block cost should have been 0.13 MB. Picks now carry their score
as a float instead of an index into the row, so nothing holds the array: the same measurement gives
1.83 MB blocked versus 4.84 MB unblocked. Block height is derived from a 64 MB target, so it adapts
instead of being a constant that is wrong at one end of the range.

The lesson generalises: a memory optimisation whose test suite only checks *results* will pass
happily while doing none of the thing it claims. The parity tests were green throughout.

**Two mutations initially survived, and both exposed real gaps in the tests.** Hoisting the
oversample window out of the per-query loop changed nothing, because the fixture was small enough
that the window covered the whole table — a purpose-built crowded vault now makes the width
observable, with a guard test asserting the fixture still distinguishes the two widths. Removing the
self-note filter changed nothing either, because the parity tests compare two paths that *share* the
assembly helper: a bug in that shared code passes both sides. Absolute-behaviour tests (a note never
suggests itself, dedupe returns each note once, scores descend) now cover what parity cannot.

Suite after the change: **1056 passed, 6 skipped, 0 failed** · `ruff` clean · `mypy` no issues.
`tests/test_connect_sweep.py` adds 29 cases; five mutations verified red→green.

### opt. 6 — Conditional query rewrite

A follow-up used to be rewritten into a standalone search query on every turn that arrived with
history, at the cost of a full LLM round-trip before retrieval even starts. It is now only rewritten
when it actually points at the previous turn.

**Gate: `eval --retrieval-only --compare` against the recorded baseline.** This is a quality change,
not a performance one, so the acceptance criterion is Hit@k / MRR, not the clock.

| | baseline | conditional |
|---|---:|---:|
| hit@1 | 1.0000 | 1.0000 |
| MRR | 1.0000 | 1.0000 |
| rewrites skipped | 0 / 14 | 4 / 14 |
| total rewrite time | 292.9 s | 204.6 s (**−30.1%**) |

`--compare` exits 0 with 0 regressions. The −30.1% is bounded by the golden set's composition (4 of
its 14 follow-ups are self-contained); real traffic depends on how users actually phrase follow-ups.

**The heuristic.** Three signals, any of which is enough: a referential word (pronoun,
demonstrative, possessive), an opening conjunction, or fewer than five words. Biased towards
rewriting on purpose — a false positive wastes one round-trip, a false negative retrieves against an
unresolved pronoun and can lose the answer, so every borderline call goes the safe way. It scores
14/14 on the golden set's follow-ups and fires on none of the 14 root questions.

**A trap in the Spanish side.** Accent-folding to catch unaccented input turns `él` (pronoun) into
`el` (definite article), which appears in almost any Spanish sentence, and made a fully
self-contained question look like a follow-up. The accented forms are matched before folding.

**Two failures the gate caught, both introduced by this change.**

The rewrite was given its own 20 s budget, chosen without measuring. Rewrites on the slowest
configuration to hand run 14.3–42.7 s (median 19.3 s), so it killed 43% of them: hit@1 fell to 0.92
and `--compare` exited non-zero. Re-sized from the measured distribution — 30 s kills 1 in 14, 45 s
kills none — to 60 s, which keeps headroom over the worst case and is still an order of magnitude
under `request_timeout_s`.

Worse, and only visible because of the first: those timeouts opened the **shared circuit breaker**.
Five failures tripped it, and the next five rewrites were cancelled outright — but the breaker also
guards answer generation, so a tight budget on an optional call was disabling the LLM for
everything. `LLMRouter.complete(optional=True)` now marks calls whose failure is not evidence of an
unhealthy backend; they neither open the breaker nor are blocked by it.

**Widening the golden set came first.** It had 2 follow-ups, both requiring a rewrite, which cannot
distinguish a working heuristic from one that fires indiscriminately. It now has 14, split between
questions that *need* resolution and questions that are already self-contained, with the
distinction documented in the file's own header.

**The gate's power is limited, and worth knowing.** Disabling the rewrite entirely drops hit@1 from
1.0 to 0.92 — only 2 of 25 positive turns depend on it. On a 10-note vault of disjoint topics a
follow-up lands correctly on a single topical word; only questions with no content words at all
("Which company created it?") actually need the rewrite to retrieve. The gate catches a
catastrophic heuristic, not a subtle one.

Suite after the change: **1106 passed, 6 skipped, 0 failed** · `ruff` clean · `mypy` no issues.
`tests/test_conditional_rewrite.py` adds 47 cases; four mutations verified red→green.

### opt. 9 — Retrieval filters

`ask` and `/api/search` can now be narrowed to a subset of the vault by category, tag or format.
`resolve_note_filter` turns the criteria into a note-id set in one query, which is threaded through
`find_hybrid`, `find_similar_notes` and `fts_search`.

**Gate 1 — the filter is respected.** Retrieval returns nothing outside the set, verified on the
real vault: unfiltered, a "biology and composition" query ranks the frogs note first; with
`--tag apple-chemistry` it returns the apples note and nothing else.

**Gate 2 — an empty match is not "no filter".** `None` means search everything; an empty set means a
filter was asked for and matched nothing, which must return nothing. Silently searching the whole
vault there would answer from notes the caller explicitly excluded. The CLI says so and stops, the
API returns an empty result, and a malformed filter is a 400 rather than an unfiltered search.

**Gate 3 — the filter must not cost latency.** It took two redesigns to get there.

The first version scored every row and masked the losers. That is the obvious implementation and it
is the wrong one: masking still multiplies every vector in the vault. Measured on 18k chunks with a
filter selecting 5% of notes, it ran **48.8% slower** than no filter at all -- a narrowing feature
that makes things slower.

The second scored only the selected rows. Better where it matters, but a filter keeping 90% of notes
copies almost the whole matrix out of itself through fancy indexing, and ran **392% slower**.

The shipped version picks by selectivity: restrict the matrix when the filter removes enough to pay
for the copy, otherwise multiply the contiguous matrix once and blank the losers. 18k chunks, 2000
notes, median of 80 runs after warm-up:

| Filter | Latency | vs unfiltered |
|---|---:|---:|
| none | 13.76 ms | — |
| 1% of notes | **1.92 ms** | **−86.1%** |
| 5% | 6.55 ms | −52.4% |
| 10% | 5.51 ms | −60.0% |
| 25% | 12.11 ms | −12.0% |
| 50% | 13.68 ms | −0.6% |
| 90% | 10.78 ms | −21.6% |

Faster across the whole range, and dramatically so where a filter is worth using.

**No date filter, deliberately.** The design called for `after`, but the only timestamps on a note
are `last_seen` and `last_tagged`, both recording when the *scanner* touched the row. On this vault
all ten notes share a single day and differ only by the minutes the scan took, so "modified after X"
would rank by scan order. It needs a real document date on the schema first; shipping a filter that
quietly answers the wrong question is worse than not shipping it.

Repeated `--tag` is AND, not OR: narrowing is the point.

Suite after the change: **1137 passed, 6 skipped, 0 failed** · `ruff` clean · `mypy` no issues.
`tests/test_retrieval_filters.py` adds 27 cases across the resolver, both retrieval paths, FTS and
the HTTP API; four mutations verified red→green.

### opt. 7 — Concurrent tagging: measured, not shipped

The proposal was a `ThreadPoolExecutor` over the tagging phase of `scan`, on the theory that
overlapping LLM calls would cut wall-clock. Its own acceptance criteria were a measurable speed-up
on a batching backend and *no regression* on local Ollama. The second one is what decided it.

**Ollama serialises.** Measured directly against the running instance, `ministral-3:8b`:

| Test | Result |
|---|---|
| 4 concurrent `/api/generate` | 20.36 s vs 21.7 s serial — **1.07x** |
| 4 concurrent embeddings | **1.33x** |
| `generate` + `embed` at once | 5.07 s vs 5.52 s serial — **1.09x** |

The ceiling for the whole feature on this configuration is about 8%, and less than that in a real
scan where parsing and disk writes also take turns.

**What it would have cost.** The tagging loop (`cli.py:228-380`) is the code that writes to the
user's notes and makes the pre-change git commits. Splitting it into prepare/commit phases is an
invasive change to the most safety-critical path in the program — and, given the measurement, it
would have shipped disabled by default. Concurrent code that neither the user nor CI ever exercises
is a liability, not a feature.

So the concurrency is deliberately not implemented. This is the one item of the ten where the work
does not pay for itself. The measurement is recorded here rather than in a commit message so that
anyone running vLLM or an OpenAI-compatible server with continuous batching — where the premise
*does* hold — can pick it up knowing exactly what was and was not tested.

**One change did come out of it.** `LLMRouter`'s breaker counters are shared state on a router the
daemon and the API use simultaneously, and `_record_failure` is a check-then-act. It now takes a
lock.

That lock is **not** a fix for an observed bug, and the first version of this entry said it was.
Checked before claiming it: a GIL build lost no increments across 160,000 concurrent `+= 1` calls
with the switch interval forced to 1 ns, and the accompanying tests pass with the lock removed. It
is forward-compatibility for free-threaded builds, on a path that already makes network calls, and
the tests say so rather than pretending to demonstrate a race they cannot produce.

---

## 9. Harness limitation: timings are not comparable across sessions

Discovered while verifying opt. 1, and it affects **every** remaining optimisation.

Comparing `connect_s` after opt. 1 against the §4 baseline gave **+19.9%**, an apparent regression.
Measuring both variants back to back in one session gives **−37.7%**. The difference is not in the
code: the same pre-opt.1 code yields `connect_s` = 26.46 s in the baseline session and **50.89 s**
in the A/B experiment session — nearly double.

The cause is the machine: an i7-6500U is a 2-core mobile chip that throttles thermally under
sustained load. And opt. 1's own success changes the workload's thermal profile: the scan used to
spend 476 s waiting on `fsync` with a cool CPU, and now does the same work in 65 s of dense CPU, so
`connect` starts on an already-hot processor.

**Rule for the remaining optimisations:**

- **Counters** (`*_db_opens`, `*_db_queries`, `load_dense_peak_mb`, `*_rows`) are deterministic and
  comparable across sessions. The §4 tables remain valid for them.
- **Timings** (`scan_s`, `connect_s`, `*_total_s`) are only valid compared **A/B within one
  session**: save the previous file, measure, apply the change, measure again. The §4 table is an
  order of magnitude, not a reference for a delta.

### Two more harness cautions

- **Statement tracing distorts timings.** The callback costs ~1.2 µs per statement and one `ask`
  executes >17,000, i.e. ~20 ms on a 100 ms measurement — enough to report the instrumentation as a
  regression. That is why `--count-queries` is opt-in: counting runs and timing runs are separate
  passes, exactly like `tracemalloc` in `bench_load_dense`.
- **The synthetic vault has a 50-word vocabulary**, so every note contains nearly the whole lexicon
  and an FTS query matches *every* chunk. That inflates FTS5's internal statements (`bm25()` reads
  one `docsize` row per matching document: 1,535 of an `ask`'s 1,552 statements are internal;
  Grimore's own are ~17). It invalidates no before/after comparison, since the vault is constant,
  but it matters for **opt. 9**, where filter selectivity is precisely what is being measured.

---

## 10. Next step

The optimisation pass is finished. Eight of the ten shipped (opts. 1, 2, 3, 4, 5, 6, 8, 9);
opt. 7 was measured and rejected, with the numbers in §8; opt. 10 (typing and coverage) ran
alongside every step rather than as an item — `mypy` gates `grimore.memory.*` and
`grimore.utils.*`, and the suite grew from 907 to 1141 tests.

What is worth doing next is not on this list, and comes out of what the work exposed:

1. **A document date on the schema.** Its absence is what blocked the `after` filter in opt. 9:
   the only timestamps record when the scanner ran, so anything time-based sorts by scan order.
2. **A larger, denser eval vault.** The gate for opt. 6 can only catch a catastrophic heuristic,
   because 10 notes on disjoint topics let a follow-up land correctly on one topical word. Every
   future retrieval-quality change inherits that blind spot.
3. **opt. 7 on a batching backend**, if one ever enters the picture. The premise holds there; §8
   records exactly what was and was not tested.

A reminder when measuring: the synthetic vault has a 50-word vocabulary, so most of an `ask`'s
statements are FTS5-internal, not Grimore's. The counter already separates them (`db_queries` vs
`db_queries_internal`); the gate is the former.

And a lesson from opt. 2's gate 3, applicable to everything that follows: **always include a control
metric** the change cannot possibly affect (`embed_s` served here). Without one there is no way to
tell a real gain from machine noise, and with 2–3 runs it is easy to convince yourself a 17% move is
signal when it is not.
