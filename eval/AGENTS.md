# Editing the eval golden set

`grimore_golden.yaml` is the baseline `grimore eval` scores retrieval + answer
quality against. This note explains how to add or change questions so the
metrics stay trustworthy. (The held-out `grimore_golden_v2.yaml`, if present,
follows the same rules — keep it untouched until you're measuring Phase 2.)

## The schema

```yaml
version: 1
questions:
  - id: kebab-stable-id          # required; reports key off it — never rename/reuse
    question: "Natural-language query"   # required
    category: single-hop         # stratum: single-hop | multi-hop | negative | …
    expected_sources: ["Roman Empire"]   # note titles (see matching below)
    expected_keywords: ["republic"]      # substrings the ANSWER must contain
    follow_ups:                  # optional; chained through one session
      - id: roman-warlords-followup
        question: "Which generals concentrated military power?"
        expected_sources: ["Roman Empire"]
        expected_keywords: ["marius", "sulla"]
```

Unknown keys are rejected (a typo fails loudly instead of silently dropping a
question). `id` and `question` are required.

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

## `expected_keywords` — substrings of the answer

Matched case-insensitively against the generated answer. Choose terms a correct
answer can't avoid — **proper nouns are safest** (`marius`, `cialdini`, `anura`).
Two traps:

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
decline rather than confabulate? Abstention detection is deterministic
(empty answer / refusal phrasings); pair with `--judge` for a stricter signal.

## Before you commit

1. `grimore scan` the reference vault (`./test_vault`), then run
   `grimore eval` (or `--retrieval-only` for a fast, deterministic check).
2. `pytest tests/test_eval.py` — `test_shipped_golden_loads_and_is_stratified`
   guards the schema.
3. Keep ids stable. A renamed id silently breaks history/regression comparisons.
