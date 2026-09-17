# Seed knowledge pack — engine bootstrap (v1)

Human-authored promoted learnings that give the engine a warm start. Committed to
`seeds/` in the new repo once it exists; the engine's bootstrap loads these as
promoted records before any agent runs. Nothing here was produced by the agents —
every record cites the artifact it was extracted from.

## Contents

- **learn_0001–0007 · failure-pattern** — one record per Darwin degradation
  category (`dead_code`, `defensive_overengineering`, `duplication`,
  `excessive_nesting`, `over_abstraction`, `unnecessary_dependency`,
  `verbose_patterns`; 30 cases total, counts in each record's provenance).
  Signal/detection/fix extracted from the case metadata and README of
  `nihanthnaidu007/Darwin`.
- **learn_0008–0012 · playbook / check** — operational findings measured by
  phase0 (`nihanthnaidu007/Darwin` → `phase0/runs/FINAL_REPORT.md`, 2,791
  rollout records): reflection-before-search, per-case noise floors, the
  bytecode-verifier's C-level blind spot, tests-are-not-the-gate (46% of
  passing candidates were worse than baseline), and determinism config.

## Schema

Each record matches the learning-record schema pinned in the design blueprint
(`art_bGBG4tqU`, "The learning record"): `id`, `type`, `title`,
`body{signal, failure_mode, detection, fix_strategy}`, `status`, `confidence`,
`provenance`, `counterfactual`, `created`, `promoted_at`.

Seeds carry `status: "promoted"` with a provenance note stating they were
promoted by human decision at bootstrap, not by the memory gate — the gate's
audit log will show that distinction for every future record.

## Verified warm start (2026-09-17)

Bootstrap verified against engine `main` at `7f9b20d` (PR #3) from the
`task/seed-the-engine-vE2LozCv` verification branch: a fresh `Engine` store,
`load_seeds(src/evogine/seeds)`, then load/store/audit/retrieval checks. Full
test suite green (`pytest -q`: 40 passed); the checks are also encoded in
`tests/test_bootstrap.py`, which runs in CI.

**Load result:** all 12 records load with `status: "promoted"` — ids
`learn_0001`…`learn_0012`, `confidence` 1.0, `promoted_at` 2026-09-17. Each
committed file is byte-identical (sha256) to the canonical seed pack. The
store lands exactly 12 record files — no ephemeral, no dropped — as a single
commit with per-seed audit lines, and re-opening the store rehydrates all 12
into retrieval. An ephemeral probe record proposed after bootstrap is never
returned by `retrieve()` (promoted-only holds under warm start).

```
$ python verify_warm_start.py        # fresh Engine + load_seeds(src/evogine/seeds)
loaded count: 12
loaded ids:   ['learn_0001', 'learn_0002', ..., 'learn_0012']
all promoted: True
record files in store: 12
statuses: ['promoted']
retrieved ids (k=20, tag-matching context): ['learn_0001', ..., 'learn_0012']
retrieved statuses: ['promoted']
bootstrap-load entries: 12  actor=engine  evidence=seed; status=promoted; confidence=1.0
store git log: 6d70af7 bootstrap: load 12 promoted seed record(s)
reopened engine retrieved: 12 records
ephemeral leak probe retrievable: False (all contexts)
ALL CHECKS PASSED
```

**Retrieval observed** (v1 type/tag relevance, k=5):

- Dead-code cleanup context ("…dead code cleanup… remove unreachable
  branches and unread symbols…") → `learn_0001` ranks #1. Positions 2–5
  (`learn_0002, 0004, 0008, 0009`) are one-token coincidences — stopwords
  like "the" and "before" pass the ≥3-char title-token rule — not signal.
- Benchmark-run context ("…decay-pressure benchmark… determinism
  configuration… fixed reflection budget… passing tests is not passing the
  gate") → the playbook/check records surface as the top-3: `learn_0012`,
  `learn_0008`, `learn_0011`.
- Unrelated context (no tag overlap, e.g. a PDF-upload task) → the no-match
  fallback still returns top-k by confidence; with all seeds at confidence
  1.0 the order degenerates to stopword-collision then id order (observed:
  `learn_0002, 0004, 0009, 0011, 0001`).

These ranking weaknesses are documented, not patched: v1 relevance is a
tracked open question (design blueprint `art_bGBG4tqU`, open decisions), and
embedding-based similarity is the candidate replacement owned by the engine.

**Schema friction:** none — every seed validates against the pinned
`LearningRecord` schema and loads unchanged; no `fix/seed-schema` branch was
needed.

## When the repo lands

1. ~~Commit `records/*.json` under `seeds/`~~ — done (PR #3, `7f9b20d`).
2. ~~The engine bootstrap loads promoted records into retrieval~~ — done
   (verified above, 2026-09-17).
3. Warm-vs-cold benchmark: "warm" arm runs with these seeds (plus whatever the
   engine promotes); "cold" arm suppresses retrieval entirely.
