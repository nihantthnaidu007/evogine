# Phase 3 — Compounding Benchmark Findings

## Headline

<!-- filled after the live run -->

## Design

Phase 3 tests whether phase 2's warm-start advantage **compounds** when the
warm lineage retains and promotes learnings across generation blocks.
Three cases from the phase-2 set:

| Case | Role | Blocks | Paired repeats/invocations |
|------|------|--------|----------------------------|
| drift-version-skew | proven warm discriminator (phase 2) | 3 | 4 |
| billing-day-offset | saturated control | 3 | 3 |
| audit-ring-drop | floor control | 3 | 3 |

Each repeat is one phase-2-identical both-arms `evogine run` invocation:
same config schema, executor (local subprocess), model, seed pack
(learn_0012 provenance, canonical `sha256(run_id|case_id|generation|candidate_index)[:8]`
derivation), retrieval_k=5, four candidates, temperature 0. The warm arm's
engine persists across blocks — one lineage per case, bootstrapped from the
12 promoted seed records (verified before the run to reproduce phase 2's
warm-start retrieval state exactly). The cold control opens no store.

Between blocks, a **scoped promotion pass** runs the production promotion
path (`evogine.swarm.promotion`) over the block's own records:

1. Every `baseline`/`candidate` rollout record is re-evaluated through the
   real code gate (local executor, same 10-second adapter timeout) and the
   fresh verdict + acceptance bit are cross-checked against the recorded
   history — any mismatch aborts the run as unsound evidence.
2. The gate evaluations are recorded as replayable `RecordedRun`s in the
   case's run log, ids via the production `recorded_run_id`.
3. Each new ephemeral proposal is decided by the production
   `finish_promotion` against **its own invocation's** recorded baseline and
   candidate runs.
4. The regression guard covers **newly promoted records only** — the 12 seed
   records are phase-2-validated on these very cases, and guarding them
   would multiply replay cost (the memory gate's production-default three
   repeats) without new information. This scope limitation applies to all
   promotion decisions reported here.

Phase-2 operational rules baked into the driver: invocations run
sequentially under the CLI's own bounded timeouts (no polling loops); the
budget guard projects total phase spend from actuals and halts past the
plan's stop line (~1.3M tokens); engine archives and ledgers land on disk
at every block boundary.

The driver ran a complete scripted dry run (all 30 invocations, 9 passes,
zero tokens) before any live call; its output lives in
`runs/compounding-dry/` and never mixes with live records.

## Live opt-in disclosure

Live invocations ran with the `--allow-local-live` opt-in: PR #18 refuses
executor-local execution with a live provider without it. The local
executor was the phase-2 benchmark configuration (Docker is the production
default; the phase-2 record explains the choice).

## Per-generation compounding view

<!-- per-case x generation arm outcomes; filled after the live run -->

## Pooled per-case outcomes

<!-- pooled medians, MAD floors, paired deltas; filled after the live run -->

## Determinism (learn_0012) coverage

<!-- checked / missing / mismatches; filled after the live run -->

## Token spend and latency per arm

<!-- actuals vs the phase-2 envelope; filled after the live run -->

## Structural nulls, degraded evidence, and stalls

<!-- full disclosure: stalls, degraded candidates, skipped proposals; filled after the live run -->

## Secrets hygiene

<!-- gitleaks + preflight scans; filled after the live run -->

## Verdict

<!-- filled after the live run -->
