# Phase 3 — Compounding Benchmark Findings

## Headline

**The warm-start advantage did not compound — it persisted, unchanged, and the promotion
gate is why.** On drift-version-skew, warm accepted **17/48** candidates vs cold's **6/48**
(~3x), but the advantage was flat across generations: warm 6/16 → 6/16 → 5/16 against
cold's constant 2/16. The saturated control stayed at the ceiling (billing-day-offset
12/12 both arms, every generation) and the floor control stayed at the floor
(audit-ring-drop 0/12 both arms, every generation) — warm was never worse anywhere, and
no regression was manufactured on either control.

The mechanism of the null is structural: all nine scoped promotion passes made **11
decisions, promoted 0, demoted 0** (134,710 replay tokens), so the warm store never grew
beyond its 12 bootstrap seeds. With a static store, retrieval injected a constant payload
— warm's token premium was **+6.4%** (465,551 vs 437,411), *smaller* than phase 2's +9.4%,
consistent with no store growth. Phase 3 therefore measures **retention** of the warm
advantage across generations, not growth-based compounding: the binding constraint on
compounding in evogine is the promotion gate's approval rate, not retrieval persistence.

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

Accepted candidates per generation block (paired both-arms invocations, 4 candidates per
arm per repeat; drift has 4 repeats per block = 16 candidates/arm, controls 3 = 12):

| Case | Role | Gen 1 (warm/cold) | Gen 2 (warm/cold) | Gen 3 (warm/cold) | Trajectory |
|------|------|-------------------|-------------------|-------------------|------------|
| drift-version-skew | warm discriminator | 6/16 vs 2/16 | 6/16 vs 2/16 | 5/16 vs 2/16 | warm flat at ~3x; no growth, no decay |
| billing-day-offset | saturated control | 12/12 vs 12/12 | 12/12 vs 12/12 | 12/12 vs 12/12 | ceiling; warm ties cold exactly |
| audit-ring-drop | floor control | 0/12 vs 0/12 | 0/12 vs 0/12 | 0/12 vs 0/12 | floor; both arms fail identically |

The compounding signature — warm's per-generation margin growing over cold — is absent.
Equally important, the phase-2 advantage did **not** decay with generation distance: the
warm arm retrieved from the same persistent store in generation 3 with the same hit rate
as generation 1. Paired warm-minus-cold deltas on drift (one per candidate pair, n=48):
15 positive, 4 negative, 29 zero — net +11, stable in sign across all three blocks.

## Pooled per-case outcomes

| Case | Arm | Candidates | Accepted | Median outcome | MAD floor | Never-worse rejections |
|------|-----|-----------|----------|----------------|-----------|------------------------|
| drift-version-skew | warm | 48 | **17** | 1.0 | 0.7413 | 0 |
| drift-version-skew | cold | 48 | 6 | 0.0 | 0.0 | 0 |
| billing-day-offset | warm | 36 | 36 | 4 | 0.0 | 0 |
| billing-day-offset | cold | 36 | 36 | 4 | 0.0 | 0 |
| audit-ring-drop | warm | 36 | 0 | 0 | 0.0 | 36 |
| audit-ring-drop | cold | 36 | 0 | 0 | 0.0 | 36 |

Warm's drift advantage is strict and large (median accepted candidate 1 vs 0), but its
MAD floor (0.74) is carried by warm's repeat variance (0–3 accepted per invocation) —
the same variance shape phase 2 saw, not a generational effect. On the floor control,
all 36 candidates from both arms were rejected by the never-worse gate: with no
improvements available, the fail-closed gates rejected everything, identically in both
arms — the no-regression requirement held with zero exceptions across all 144
non-drift candidates (72 per arm).

## Determinism (learn_0012) coverage

**480 candidate/verifier records checked, 0 missing determinism blocks, 0 seed mismatches**
(240 candidate + 240 verifier across 30 invocations; phase 2 checked 560 across 35).
Every live record reproduces the canonical `sha256(run_id|case_id|generation|candidate_index)[:8]
derivation re-verified against learn_0012 before the first live call. The determinism
chain survived all three cases, all nine promotion passes' replayed runs, and the
pre-run seed-pack equivalence check (fresh bootstrap vs the phase-2 engine: same 12
promoted IDs, identical retrieval at k=5 for all three cases).

## Token spend and latency per arm

| Arm | Model calls | Input tokens | Output tokens | Total | Wall latency (summed per-call) |
|---|---|---|---|---|---|
| cold | 257 | 276,247 | 161,164 | 437,411 | 2,932.6 s |
| warm | 251 | 298,962 | 166,589 | 465,551 | 3,009.4 s |

Warm's premium is **+28,140 tokens (+6.4%)** — smaller than phase 2's +9.4%, consistent
with a store that never grew (retrieval injected the same 12-seed payload every
invocation). Call counts: 120 candidate + 120 verifier per arm; reflectors fired 17
times cold vs 11 warm (the reflector runs only when the builder emits learnings).
Baselines (60 calls) are replayed with zero tokens and excluded from the arm rows.
Stage split (pooled): candidates 240 calls (230,928 in / 128,571 out), verifiers 240
(321,763 in / 186,931 out), reflector 28 (22,518 in / 12,251 out) — 902,962 invocation
tokens, matching the per-arm sums exactly. Promotion replay added 134,710 tokens
across the nine passes (drift 41,645; billing 0; audit 93,065).

**Total phase spend: 1,037,672 tokens** — 79.8% of the plan's 1.3M stop line, ~4% above
the ~1M working envelope (the promotion-pass replay — 134.7K across 9 passes — has no
phase-2 analogue; per-invocation cost, ~30.1K vs phase 2's ~26.0K, reflects drift's
4-repeat blocks and pass overhead, not a per-unit regression). No per-case stop line was
breached: drift finished at 519,146 of its 520K line (99.8% — the tightest case, as
projected), billing at 231,399 of 390K, audit at 287,127 of 390K. Accounting note: the
per-case `spend-*.json` snapshots are written at block boundaries and lag each case's
final promotion pass, so summing those files undercounts (983,286); the authoritative
total is consolidated-rollout tokens (902,962) plus the promotion ledger's replay count
(134,710), which reconcile to the per-case invocation sums above.

## Structural nulls, degraded evidence, and stalls

**Structural null — zero promotions.** Nine scoped promotion passes produced 11 decided
proposals, 0 promoted, 0 demoted. The production counterfactual gate approved no
ephemeral learning, so the warm store stayed at its 12 bootstrap seeds for the entire
phase. This is the load-bearing null: with nothing added to the store, "compounding"
reduces to retention, which is what the data shows. The scoped regression guard (newly
promoted records only) consequently never guarded anything; no skipped proposals exist
— every queued proposal was decided.

**Degraded evidence: none.** Zero degraded markers across all 30 invocation reports. No
provider faults, no bounded-timeout degradations (the PR #29 ladder never fired), no
rate limits, no retried-with-error model calls, no partial records, no rerun utility
use.

**Stalls: none in the run.** All 30 invocations ran sequentially to completion and all
nine passes decided. One operator-level disclosure: the coordinating worker exhausted
its platform wake budget twice mid-run on fallback timers (phase 2's failure mode was
self-timer polling; this run used the detached watchdog instead). The benchmark was
never touched — both tmux sessions ran uninterrupted, no invocation was restarted or
duplicated, and all evidence through block 6 was checkpoint-pushed before the revival.

**Live opt-in:** all 30 invocations and all promotion replay ran executor-local with
the live provider under the `--allow-local-live` opt-in (see Design).

## Secrets hygiene

Gitleaks passed in CI at every pushed head, including the checkpoint tip `1efbf84` and
this report's final commit. The API key was supplied via environment variable only;
no credential appears in the configuration, findings, engine archives, or rollout
records (rollout prompts carry case content and retrieved learning text, nothing
else). No manual secret handling occurred beyond the pre-run preflight scans.

## Verdict

**The compounding hypothesis is not supported under production promotion semantics —
and the null is now well explained.** The warm-start advantage is durable: it survived
three generation blocks unchanged (~3x acceptance on drift, exact ties on both
controls, never worse anywhere, 480/480 determinism clean). But it did not grow,
because the promotion gate promoted none of the 11 proposals it decided. With a static
store, warm's cost premium shrank (+6.4% vs phase 2's +9.4%) — a static store injects
a constant payload.

The phase-2 conclusion stands and strengthens: warm persistence is safe (never worse)
and strictly better exactly where retrieval applies. The binding constraint on
compounding in evogine is the promotion gate's approval rate, not retrieval
persistence. Any longer-horizon phase should first study that gate (11 decisions, 0
promotions) before buying more generations — additional blocks over a static store
would re-measure retention at linear cost.
