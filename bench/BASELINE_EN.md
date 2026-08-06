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
| `scan_s` | **476.09 s** | 474.94 | 477.25 | 0.48% | ✅ | B1 · opt. 1, 3 |
| `scan_db_opens` | **66,396** | 66,396 | 66,396 | 0.00% | ✅ | **B1 · opt. 1** |
| `connect_s` | **26.46 s** | 26.37 | 26.54 | 0.63% | ✅ | B4 · opt. 5 |
| `connect_db_opens` | **12,003** | 12,003 | 12,003 | 0.00% | ✅ | B4 · opt. 5 |
| `load_dense_s` | **0.1122 s** | 0.1103 | 0.1141 | 3.39% | ✅ | B3 · opt. 4 |
| `load_dense_peak_mb` | **121.22 MB** | 121.22 | 121.22 | 0.00% | ✅ | **B3 · opt. 4** |
| `load_dense_rows` | 17,465 | — | — | 0.00% | ✅ | context |
| `load_dense_matrix_mb` | 53.65 MB | — | — | 0.00% | ✅ | context |

### 4.2 `ask` with `top_k=5` — warm and cold

Measured separately because they are the **two real callers**. A one-shot `grimore ask` builds the
dense matrix inside its only query, so its `retrieve_s` carries the full `load_dense` cost and
inherits its variance. The shell keeps a `Session` alive, so every query after the first hits the
connector's signature-sealed cache. The Oracle's N+1 (B2, opt. 2) lives in the warm path; measuring
them together measured neither.

| Metric | Mean | Spread | ±5% | Note |
|---|---:|---:|:---:|---|
| `warm.total_s` | **0.1010 s** | 0.68% | ✅ | shell / live `Session` |
| `warm.retrieve_s` | **0.0894 s** | 0.35% | ✅ | **opt. 2 gate** |
| `warm.embed_s` | 0.0009 s | 1.03% | ✅ | embedding cache |
| `warm.db_opens` | **13.0** | 0.00% | ✅ | **opt. 2 gate** (target: ≤2) |
| `warm.generate_s` | 0.0026 s | 12.81% | ❌ | 2.6 ms of stub; no opt. touches it |
| `cold.total_s` | 0.2289 s | 16.59% | ❌ | one-shot CLI |
| `cold.retrieve_s` | 0.2057 s | 18.06% | ❌ | dominated by matrix build |
| `cold.db_opens` | **15.0** | 0.00% | ✅ | +2 over warm: matrix load |

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

### opt. 8 — Magic constants → configuration ✅

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

### opt. 1 — Reusable per-thread SQLite connection ✅

One connection per thread, alive for the lifetime of the `Database`, with the PRAGMAs and the
sqlite-vec load applied **once** instead of on each of the 73 data-access paths. Adds
`busy_timeout=5000`, `cache_size=-16000` and `mmap_size=256MB` (useless before: a cache discarded
microseconds later never hits). Explicit teardown via `Database.close()`, called from
`Session.close()` and `daemon.stop()`.

| Metric | pre-opt.1 | opt.1 | Change | Criterion |
|---|---:|---:|---:|---|
| `scan_db_opens` | 66,396 | **1** | −100% | O(1) per thread ✅ |
| `connect_db_opens` | 12,003 | **1** | −100% | ✅ |
| `scan_s` | 529.75 s | **65.14 s** | **−87.7%** | −30% or more ✅ |
| `connect_s` | 50.89 s | **31.72 s** | **−37.7%** | (not required) ✅ |

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

### opt. 3 — Batched embedding inserts ✅

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

### opt. 2 — Killing the Oracle's N+1 ✅

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

With opt. 8 → 1 → 3 → 2 closed, the next block is an independent chain:

1. **opt. 4** — `_load_dense` without `text_content` + an `.npy` cache with `mmap` (2 d, low risk).
   Gate: `load_dense_peak_mb` drops, and the cache invalidates when the vault changes.
2. **opt. 5** — vectorised `connect`, blocked `M @ M.T` (1.5 d, low risk). Depends on opt. 4 and
   reuses the `get_note_titles(ids)` opt. 2 just introduced. Gate: suggested links above the
   threshold are **the same**.

The baseline for both is already taken: `load_dense_peak_mb` and `connect_s` in §4.1.

A reminder when measuring: the synthetic vault has a 50-word vocabulary, so most of an `ask`'s
statements are FTS5-internal, not Grimore's. The counter already separates them (`db_queries` vs
`db_queries_internal`); the gate is the former.

And a lesson from opt. 2's gate 3, applicable to everything that follows: **always include a control
metric** the change cannot possibly affect (`embed_s` served here). Without one there is no way to
tell a real gain from machine noise, and with 2–3 runs it is easy to convince yourself a 17% move is
signal when it is not.
