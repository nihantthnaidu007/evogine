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

## When the repo lands

1. Commit `records/*.json` under `seeds/`.
2. The engine bootstrap loads promoted records into retrieval.
3. Warm-vs-cold benchmark: "warm" arm runs with these seeds (plus whatever the
   engine promotes); "cold" arm suppresses retrieval entirely.
