# Editing the eval golden set

`grimore_golden.yaml` is the historical baseline that `grimore eval` uses by
default. Phase 0 uses the explicit tuning and evaluation files described below.
This note explains how to add or change questions so the metrics stay
trustworthy.

## The schema

`grimore_golden.yaml` remains the historical v1 baseline. New Phase-0 datasets
use v2 and are split into `phase0_tuning.yaml` and
`phase0_evaluation.yaml`. Never use the evaluation file while tuning.

```yaml
version: 2
set: tuning                    # tuning | evaluation
corpus_version: phase0-test-vault-2026-09-22
questions:
  - id: kebab-stable-id          # required; reports key off it — never rename/reuse
    question: "Natural-language query"   # required
    category: direct
    language: en                 # es | en
    case_type: direct            # direct | ambiguous | comparison | …
    document_formats: [md]
    user_task: find-fact         # find-fact | compare-documents | review-note
    difficulty: long-document
    split: tuning                # must match the top-level set
    expected_sources: ["Roman Empire"]   # note titles (see matching below)
    expected_keywords: ["republic"]      # terms the ANSWER must contain
    expected_answer: "A concise reference answer."
    expected_evidence:
      - source: "Roman Empire"
        passage: "An exact passage present in the extracted document."
        locator: "Fall of the Republic"
    review_criteria: "What a human reviewer must require."
    follow_ups:                  # optional; chained through one session
      - id: roman-warlords-followup
        question: "Which generals concentrated military power?"
        expected_sources: ["Roman Empire"]
        expected_keywords: ["marius", "sulla"]
        expected_answer: "Marius and Sulla."
        expected_evidence:
          - source: "Roman Empire"
            passage: "Rise of Warlord Generals: Marius & Sulla"
        review_criteria: "Must name both generals."
```

Unknown keys are rejected (a typo fails loudly instead of silently dropping a
question). `id` and `question` are required. In v2, positive cases also require
sources, answer, evidence and review criteria. Negative cases require
`expected_answer: abstain`, no evidence, and `case_type: unanswerable`.

## `expected_sources` — how matching works

Sources match **token-normalised**, not as exact strings:

- The retrieved title is lowercased, accent-folded (`Astronomía` → `astronomia`),
  emoji/punctuation-stripped, and split into alphanumeric tokens.
- Your entry matches if **all** of its tokens are present in that set.

So write a **short, distinctive** fragment — `Roman Empire`, not the note's full
decorated H1 `🏛️ The Roman Empire: An Exhaustive Historical and Structural
Analysis`. A note's title is its frontmatter `title:`, else its first `# H1`,
else its filename stem (so `gothic_architecture.txt` → `gothic_architecture`).

Pick tokens that identify **exactly one** note. `Software Architecture` is fine;
a lone `architecture` could match more than one. To see the title the system
actually prints, run `grimore search "<query>"`.

## `expected_keywords` — terms the answer must contain

Matched against the generated answer, **accent-folded** (`energía` ⇄ `energia`)
and **word-boundary anchored** on the leading edge. Practical consequences:

- Accents don't matter on either side — the LLM's accenting is non-deterministic.
- A keyword only counts at a word start, so `art` no longer matches `Sparta`;
  pick terms that stand as their own word.
- Simple inflection still counts (there's no trailing boundary): `frog` matches
  `frogs`, `orbit` matches `orbital`.
- Multi-word keywords tolerate any separator: `machine learning` matches
  `machine-learning` too.

Choose terms a correct answer can't avoid — **proper nouns are safest**
(`marius`, `cialdini`, `anura`). Two traps:

- **Language.** Ask in the note's language so the answer's language matches your
  keywords. The Spanish notes here use Spanish questions + Spanish keywords.
- **Grounding.** A keyword the source note doesn't contain can never appear in a
  RAG answer. Confirm it's in the note before adding it.

## Categories

`category` is a free-form stratum so the report breaks ranking down by query
type (and shows where retrieval struggles). Conventions in this file:

| category     | meaning                                            |
|--------------|----------------------------------------------------|
| `single-hop` | one note answers it directly                       |
| `multi-hop`  | needs synthesis / several facts (often with follow-ups) |
| `negative`   | the answer is **not** in the vault                 |

Add new categories freely — they aggregate automatically. Follow-ups inherit
their parent's category.

## Negative cases

A `negative: true` case has no answer in the vault. **Omit `expected_sources`**
(the loader rejects a negative that lists any). These are excluded from the
ranking aggregates and instead scored by `abstention_rate`: did the Oracle
decline rather than confabulate? Positive cases are also checked for
`incorrect_abstention_rate`, so a system cannot improve by refusing every
question. Abstention detection is deterministic (empty answer / refusal
phrasings); pair with `--judge` for the directional support metrics below.

## Citation metrics (Phase 0 semantics)

Do not call reference membership "claim support". The harness reports these
separately:

- `citation_validity`: fraction of emitted wikilinks that match retrieved
  sources. It is `null`, not `1.0`, when the answer has no citations.
- `cited_answer_rate`: fraction of generated positive answers that contain at
  least one citation.
- `context_evidence_recall`: literal expected-passage presence in a delivered
  context fragment at its expected physical page/heading.
- `cited_evidence_recall`: the same reference also has its fragment label and
  required locator cited in the answer. A bare title cannot earn page credit.
- `selected_evidence_recall`: an accepted occurrence unambiguously selects a
  delivered fragment containing the expected passage at its required locator.
  Short tokens retain the exact request-local index; a canonical label shared
  by several fragments is ambiguous, even if page coverage gives credit.
  Missing historical selection traces are unassessed, never reconstructed;
  an explicitly empty trace has zero selections. Denominators and gaps are
  separate from the historical page metric and `pass_rate`.
- `claim_citation_coverage`: judge-estimated fraction of factual claims with a
  nearby citation that supports them.
- `automated_claim_support`: judge-estimated fraction of verifiable factual
  claims supported by the indexed passages, whether cited or not.
- claim support: the final human-review measure against the exact expected
  passage; it is not inferred from any metric above.

The two judge-estimated metrics are directional diagnostics, never CI gates.
Generated positive cases with `review_criteria` are copied into the report's
`manual_review` queue with their exact passages, expected evidence and a
pending human verdict.

Generated negative cases with review criteria are exported separately in
`negative_manual_review`, with exact delivered passages, answer, detected cue
and `abstention_verdict: pending`. Never turn a refusal phrase or a model's
`insufficient_evidence` status into a semantic abstention verdict. A response
can refuse one detail and invent another; an empty/silent response can be a
technical failure. The isolated answer bank's `negative_control` diagnostic
excludes empty, silent, unassessed and no-context answers from its cue-check
denominator. It does not rewrite historical `abstention_rate` or `pass_rate`.
Publish that denominator and unassessed cases alongside pending review. Absence
claims in `phase4_unanswerable_answers.yaml` apply only to the copied control
PDF, never to all of `test_vault`; keep this manifest separate from the three
positive cases and from held-out evaluation.

Passage presence normalizes case, Unicode composition and whitespace only;
numbers, punctuation and accents remain significant. It never joins fragments
or searches the full corpus. Missing legacy details are unassessed (`null`);
an explicitly empty delivered context scores zero. Reports expose assessed
turns, reference counts and hits. These diagnostics do not change historical
`pass_rate`; `evidence_gaps` surfaces issues even when that old outcome is `ok`.

## Generation failures and long runs

A turn whose answer model returned nothing (server crash, missing `answer`
field, raised `ask`) records `generation_error`, skips the judges and adds to
`generation_failure_n`. Its silent sentinel still reads as an abstention cue
in the historical metrics, which are deliberately not redefined, so a report
with `generation_failure_n > 0` is contaminated: rerun those cases instead of
publishing it. Long real-model runs should pass `--checkpoint <run.jsonl>`:
finished cases are saved one by one, a rerun restores them, failed cases are
asked again, and a checkpoint from a different identity (cases, config,
corpus, code fingerprint, mode, k values, judge) is refused.

## Explicit presentation controls

The isolated Phase-4 answer bank can use `--context-order rotate-first-last`.
This fixed permutation acts only on already accepted context, not connector
ranking or expected evidence. Export original/presented indices, token maps,
passage hashes and the declared strategy; keep the actual retrieval ranking
for ranking metrics. Never promote discarded/deeper candidates to improve a
score or select an order using reference answers. Default `ranked` does not
alter the ordinary presentation. A few passages of one controlled document
are not representative distractor documents or a robustness benchmark. Keep
all repetitions and failures, with human claim support still pending.

## Before you commit

The optional pre-generation sufficiency check is a separate selection policy,
not the fixed presentation permutation. It may reduce generation context only
to model-selected, literally verified excerpts of already accepted passages.
Keep the real retrieval ranking and pre-check candidate records, selected
quote provenance, reference maps, context hashes, decision, failures and extra
call cost. Literal quote provenance is never semantic sufficiency. Unassessed
checks must not receive abstention credit or expected-evidence delivery credit.
Test answerable cases and negatives separately, and keep every failed run.

The opt-in cited-claims protocol renders only refs explicitly emitted by the
answer model for each claim. Never borrow checker refs to fill missing answer
refs, append guessed citations, partially release malformed claims, or treat
format validation as semantic support. Preserve bounded untrusted parsed-model
snapshots (including rejected proofs), truncation flags and per-stage failures.
Protocol v3 reports and their fingerprints are distinct from historical v2
measurements. The acceptance runner plans by default; execute only when asked
to measure, using fresh processes and a new output directory. Paired comparisons
must match the pre-selection context, ranking, model identities, code and settings
apart from the declared switch; keep failures in counts and human verdicts pending.

1. `grimore scan` the reference vault (`./test_vault`), then run
   `grimore eval` (or `--retrieval-only` for a fast, deterministic check).
2. `pytest tests/test_eval.py` — `test_shipped_golden_loads_and_is_stratified`
   guards the schema.
3. Keep ids stable. A renamed id silently breaks history/regression comparisons.
