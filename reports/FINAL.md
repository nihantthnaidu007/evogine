# Evogine Benchmark Program — Final Report

Status: **complete**. This report consolidates the three merged phase reports —
[PILOT.md](PILOT.md), [CROSSCASE.md](CROSSCASE.md), [COMPOUNDING.md](COMPOUNDING.md) —
into one document. No new runs were performed and no live API calls were made: every
number below is quoted or computed from the committed phase reports and the raw
evidence under `runs/pilot/`, `runs/crosscase/`, `runs/compounding/`, and
`runs/compounding-dry/`. Program pins: phase-1 report pin `019866a` (merged as PR
#9, `ec3b994`); phase 2 ran on main `74f707b` (merged as PR #24, `acd8435`);
phase 3 merged as PR #30 (`6a2af31`) — this report's base.

## 1. Executive verdict

**Warm-start retrieval is never worse than cold-start; where it helps, the help is
real but narrow — and under v1's promotion gate it does not grow.**

| Claim | Evidence |
|---|---|
| Warm is **never worse** than cold | Warm ≥ cold in every paired comparison across all 67 paired both-arms invocations (2 pilot + 35 cross-case + 30 compounding); zero manufactured regressions on all 144 non-drift phase-3 candidates (72 per arm, exact ties on both controls) |
| Strict advantage on **one case only** | `drift-version-skew`: phase-3 pooled warm **17/48** accepted vs cold **6/48** (~3×); median outcome 1.0 vs 0.0; the only case whose paired delta (net +11: 15 positive / 4 negative / 29 zero) clears the learn_0009 noise floor (1.4826 × MAD = 0.7413) |
| The advantage **persisted** across generations | Warm 6/16 → 6/16 → 5/16 vs cold 2/16 → 2/16 → 2/16 — flat at ~3×, no decay with generation distance |
| It **did not compound** | Nine production promotion passes decided **11 proposals and promoted 0** (0 demoted); the warm store never left its 12 bootstrap seeds, so retrieval injected a constant payload (warm premium +9.4% in phase 2 → +6.4% in phase 3) |

**Verdict for a newcomer:** v1 measures **retention, not growth**. The engine safely
persists a warm-start advantage across generation blocks, and the fail-closed gates
never let it get worse. But the promotion gate approved nothing, so no compounding
could occur: the binding constraint on compounding in evogine is the promotion gate's
approval rate, not retrieval persistence. Program cost: **1,999,652 tokens** (~2.0M)
across the three phases, inside every per-case and per-phase stop line.

## 2. Program design at a glance

| Phase | Question | Scale | Merged |
|---|---|---|---|
| 1 — Pilot | Does the loop validate end to end with a real model, and what does warm vs cold look like at minimum scale? | 2 cases × both arms × 1 paired run × 4 candidates | PR #9 (`ec3b994`) |
| 2 — Cross-case | Does the warm advantage generalize across the 10-case corpus? | 10 cases × both arms × 4 paired repeats — 35 of 40 planned invocations completed | PR #24 (`acd8435`) |
| 3 — Compounding | Does the warm advantage compound across generation blocks under production promotion? | 3 cases × 3 blocks × (4/3/3 paired repeats) = 30 invocations + 9 promotion passes | PR #30 (`6a2af31`) |

Shared protocol, identical in every phase: paired both-arms invocations sharing one
`run_id`, so `(generation, candidate_index)` seeds match across arms and only retrieval
differs; per-record seeds `sha256(run_id|case_id|generation|candidate_index)[:8]`
(learn_0012); temperature 0.0, thinking off; `retrieval_k=5`, 4 candidates per
generation; noise floor per learn_0009 = **1.4826 × MAD** of paired repeats, with
sub-floor deltas labeled *indistinguishable from noise*; provider Anthropic
`claude-sonnet-4-5`; executor `local` (LocalSubprocessSandbox — no Docker in the
benchmark environment, a disclosed residual, see §9); budget guard with per-phase and
per-case stop lines; and a scripted zero-token dry run of the phase-3 driver (all 30
invocations, 9 passes) before any live call.

## 3. Phase 1 — pilot (PR #9)

Full report: [PILOT.md](PILOT.md).

**Design.** 2 cases — `audit-ring-drop` (single-function golden_io scope) and
`concurrency-account-ledger` (package scenario scope) — × both arms × 1 paired run ×
4 candidates. Purpose: validate the full pipeline live (harness → swarm runner
builder/verifier/gate → sandboxed code gate via local executor → retention → reflector
→ rollout records) and size the token budget for phases 2–4.

**Results.** 38/38 rollout records carry the full phase0 schema plus the learn_0012
determinism block (32/32 candidate+verifier records checked, 0 missing). Warm retrieval
fired correctly — the warm arm retrieved seeded records, cold retrieved none, no prompt
contamination. Outcomes were binary and uninformative by design:

| Case | Outcome | Reading |
|---|---|---|
| `audit-ring-drop` | 0/8 accepted; all 8 learn_0011 never-worse rejections | Degraded baseline and all eight repairs measure 44 bytecode ops; retention requires strict improvement. The gate is doing its job; the case cannot discriminate (measured structural tie) |
| `concurrency-account-ledger` | 8/8 accepted — 4/4 in **both** arms | Solvable with and without retrieval at this difficulty; paired deltas all zero |

All 8 paired deltas are exactly zero — zero variance — so the pilot's binary outcome is
**indistinguishable from noise / structural tie** under the learn_0009 floor. Power
conclusion: with zero observed variance, no repeat count resolves a nonzero delta
better; phases 2–3 therefore keep the 4-paired-repeat floor (repeats are not multiplied
without a variance-driven reason) and report per-case medians + MAD floors over
run-level outcome scores.

**Found live and fixed:** (1) fenced ```` ```json ```` package replies broke
`extract_code` — 8/8 live replies fenced; fixed with a tag-agnostic regex plus
defensive fence strip and regression tests; (2) collapsed-candidate retention semantics
confirmed deliberate via the #14 contract test — a candidate with no re-evaluable gate
evidence must not reach the reflector (structural blind spot: the reflector never
diagnoses builder-level extraction failures); (3) bridge latency dropped converting
bench↔swarm protocols (fixed at `c538b4f`).

**Pilot token spend** (38 records; the budget-sizing basis for phases 2–4):

| Stage | Calls | Input tokens | Output tokens |
|---|---|---|---|
| baseline | 4 | 0 (scripted) | 0 |
| candidate (builder) | 16 | 13,608 | 6,170 |
| verifier | 16 | 17,521 | 10,503 |
| reflector | 2 | 1,274 | 901 |
| **total** | **34** | **32,403** | **17,574** |

Total **49,977 tokens**. Per-arm per-generation unit ≈ **16.2K in + 8.8K out** for a
4-candidate, 1-generation case run. Verifiers dominate (~60% of latency, ~54% of
output tokens); wall-clock ≈ 3 min per case. Secrets: 0/38 records tainted.

## 4. Phase 2 — cross-case (PR #24)

Full report: [CROSSCASE.md](CROSSCASE.md).

**Design.** 10 cases × (warm, cold) × 4 paired repeats on main `74f707b` (post-#29
bounded timeouts, #28 fail-closed degradation, #14 promotion wiring, and the
oracle-guard / staging-escape / env-allowlist hotfixes).

**Per-case outcomes** (medians over run-level outcome scores; per-run paired
warm−cold delta vectors live in `runs/crosscase/results.csv`):

| Case | Arm | Candidates | Accepted | Median outcome | MAD floor | Never-worse rejections |
|---|---|---|---|---|---|---|
| audit-ring-drop | cold | 16 | 0 | 0.0 | 0.0 | 16 |
| audit-ring-drop | warm | 16 | 0 | 0.0 | 0.0 | 16 |
| billing-day-offset | cold | 16 | 16 | 4.0 | 0.0 | 0 |
| billing-day-offset | warm | 16 | 16 | 4.0 | 0.0 | 0 |
| concurrency-account-ledger | cold | 16 | 16 | 4.0 | 0.0 | 0 |
| concurrency-account-ledger | warm | 16 | 16 | 4.0 | 0.0 | 0 |
| drift-version-skew | cold | 16 | 2 | 0.5 | 0.7413 | 0 |
| drift-version-skew | warm | 16 | 6 | 1.5 | 0.7413 | 0 |
| load-spike-shedder | cold | 16 | 16 | 4.0 | 0.0 | 0 |
| load-spike-shedder | warm | 16 | 16 | 4.0 | 0.0 | 0 |
| partial-batch-write | cold | 16 | 14 | 4.0 | 0.0 | 0 |
| partial-batch-write | warm | 16 | 15 | 4.0 | 0.0 | 0 |
| retry-storm-budget | cold | 16 | 16 | 4.0 | 0.0 | 0 |
| retry-storm-budget | warm | 16 | 16 | 4.0 | 0.0 | 0 |
| tz-deadline-rollover | cold | 16 | 16 | 4.0 | 0.0 | 0 |
| tz-deadline-rollover | warm | 16 | 16 | 4.0 | 0.0 | 0 |
| unicode-nfkc-key | cold | 12 | 0 | 0.0 | 0.0 | 12 |
| unicode-nfkc-key | warm | 12 | 0 | 0.0 | 0.0 | 12 |
| counter-overflow-wrap | — | 0 | 0 | structural null | — | — |

Reading the table: of the 9 measured cases (counter-overflow-wrap is a structural
null), **7 have exactly zero paired deltas** — including the two floor cases
(`audit-ring-drop`, `unicode-nfkc-key`), where both arms accept zero and no retrieval
effect is possible — and 5 saturate at the 4.0 ceiling where retrieval cannot matter.
`partial-batch-write` leans warm (15 vs 14 accepted) but stays under its floor.
`drift-version-skew` is the **only** case that clears its noise floor.

**The one discriminating case.** Warm's candidates passed the objective gate on 6/16
attempts against cold's 2/16, with a median paired delta (+1.0) above the 0.7413 MAD
floor. This is directionally coherent with the engine thesis — version-skew is exactly
the failure class where remembering how a prior version broke should help — but at
n=4 paired repeats the honest summary is: a positive, noise-floor-clearing signal on
one of nine cases, no evidence of harm anywhere, and no evidence of broad warm
advantage at this sample size.

**Determinism.** 560 candidate/verifier records verified against the canonical seed
derivation: 0 missing playbook blocks, 0 seed mismatches.

**Token spend and latency:**

| Arm | Model calls | Input | Output | Total | Latency (summed per-call) |
|---|---|---|---|---|---|
| cold | 289 | 272,987 | 162,456 | 435,443 | 3,186.8 s |
| warm | 287 | 305,929 | 170,631 | 476,560 | 3,387.9 s |

Combined **912,003 tokens** across 35 completed invocations, ~26.0K each — matching
the pilot's per-invocation projection, so repeats were held at the 4-per-arm floor.
Warm's premium is **+41,117 tokens (+9.4%)**, the price of injecting retrieved
learnings. Stage split (pooled): candidates 280 calls (245,194 in / 116,941 out),
verifiers 280 (320,840 in / 209,739 out), reflector 16 (12,882 in / 6,407 out). Known
reporting gap: the 70 baseline records carry no token fields, so baseline spend is not
represented. Secrets: 0/657 records tainted. Structural nulls and the phase's
incidents: §9. **Verdict: no harm, targeted help.**

## 5. Phase 3 — compounding (PR #30)

Full report: [COMPOUNDING.md](COMPOUNDING.md).

**Design.** Three cases from the phase-2 set — `drift-version-skew` (proven warm
discriminator; 4 paired repeats per block), `billing-day-offset` (saturated control;
3), `audit-ring-drop` (floor control; 3) — × 3 generation blocks = 30 paired
both-arms invocations. The warm arm's engine persists across blocks (one lineage per
case, bootstrapped from the 12 promoted seed records; verified before the run to
reproduce phase-2's warm-start retrieval exactly — same 12 promoted IDs, identical
retrieval at k=5). Between blocks, a **scoped promotion pass** runs the production
promotion path over the block's own records: every baseline/candidate record
re-evaluated through the real code gate with verdicts cross-checked against recorded
history (any mismatch aborts as unsound), gate evaluations recorded as replayable
`RecordedRun`s, each new ephemeral proposal decided by the production
`finish_promotion` against its own invocation's runs, and a regression guard scoped to
**newly promoted records only** (the 12 seeds are phase-2-validated on these very
cases). The driver ran a complete scripted dry run (30 invocations, 9 passes, zero
tokens) before any live call; its output lives in `runs/compounding-dry/`.

**Per-generation compounding view** (accepted candidates per block, warm vs cold):

| Case | Role | Gen 1 | Gen 2 | Gen 3 | Trajectory |
|---|---|---|---|---|---|
| drift-version-skew | warm discriminator | 6/16 vs 2/16 | 6/16 vs 2/16 | 5/16 vs 2/16 | warm flat at ~3×; no growth, no decay |
| billing-day-offset | saturated control | 12/12 vs 12/12 | 12/12 vs 12/12 | 12/12 vs 12/12 | exact ceiling ties |
| audit-ring-drop | floor control | 0/12 vs 0/12 | 0/12 vs 0/12 | 0/12 vs 0/12 | exact floor ties |

The compounding signature — warm's per-generation margin growing over cold — is
absent. Equally important, the phase-2 advantage did **not** decay with generation
distance: warm retrieved from the same persistent store in generation 3 with the same
hit rate as generation 1. Paired drift deltas (one per candidate pair, n=48): 15
positive, 4 negative, 29 zero — net +11, stable in sign across all three blocks.

**Pooled per-case outcomes:**

| Case | Arm | Candidates | Accepted | Median outcome | MAD floor | Never-worse rejections |
|---|---|---|---|---|---|---|
| drift-version-skew | warm | 48 | **17** | 1.0 | 0.7413 | 0 |
| drift-version-skew | cold | 48 | 6 | 0.0 | 0.0 | 0 |
| billing-day-offset | warm | 36 | 36 | 4 | 0.0 | 0 |
| billing-day-offset | cold | 36 | 36 | 4 | 0.0 | 0 |
| audit-ring-drop | warm | 36 | 0 | 0 | 0.0 | 36 |
| audit-ring-drop | cold | 36 | 0 | 0 | 0.0 | 36 |

Warm's drift advantage is strict and large, but its MAD floor (0.7413) is carried by
warm's own repeat variance (0–3 accepted per invocation) — the phase-2 variance shape,
not a generational effect. On the floor control, fail-closed gates rejected all 36
candidates in both arms; across all 144 non-drift candidates (72 per arm), the
no-regression requirement held with zero exceptions.

**The promotion null (the load-bearing result).** Nine scoped promotion passes made
**11 decisions, promoted 0, demoted 0** (`runs/compounding/promotion-passes.jsonl`;
the `decided/` ledgers record 9 audit + 2 drift proposals, every decision `ephemeral`,
none skipped). The production counterfactual gate approved no ephemeral learning, so
the warm store stayed at its 12 bootstrap seeds for the entire phase; the scoped
regression guard consequently never guarded anything.

**Determinism.** 480/480 candidate/verifier records clean (240 candidate + 240
verifier across 30 invocations); the chain survived all nine passes' replayed runs and
the pre-run seed-pack equivalence check.

**Token spend and latency:**

| Arm | Model calls | Input | Output | Total | Latency (summed per-call) |
|---|---|---|---|---|---|
| cold | 257 | 276,247 | 161,164 | 437,411 | 2,932.6 s |
| warm | 251 | 298,962 | 166,589 | 465,551 | 3,009.4 s |

Warm's premium is **+28,140 tokens (+6.4%)** — smaller than phase 2's +9.4%,
consistent with a store that never grew (retrieval injected the same 12-seed payload
every invocation). Call counts: 120 candidate + 120 verifier per arm; reflectors fired
17 times cold vs 11 warm; baselines (60 calls) replayed with zero tokens. Stage split
(pooled): candidates 240 calls (230,928 in / 128,571 out), verifiers 240 (321,763 in /
186,931 out), reflector 28 (22,518 in / 12,251 out) — **902,962 invocation tokens**.
Promotion replay added **134,710 tokens** across the nine passes (drift 41,645;
billing 0; audit 93,065). **Total phase spend: 1,037,672 tokens — 79.8% of the plan's
1.3M stop line**, ~4% above the ~1M working envelope (the promotion-pass replay has no
phase-2 analogue; per-invocation ~30.1K vs phase 2's ~26.0K reflects drift's
4-repeat blocks and pass overhead, not a per-unit regression). No per-case stop line
breached: drift 519,146 of 520K (99.8%), billing 231,399 of 390K, audit 287,127 of
390K. **Degraded evidence: none** — zero degraded markers across all 30 invocation
reports; no provider faults, no bounded-ladder fires, no rate limits, no partial
records. Incidents: §9. **Verdict:** the compounding hypothesis is not supported under
production promotion semantics — and the null is now well explained.

## 6. Cross-phase token spend

| Phase | Model calls | Total tokens | Stop line | Warm premium |
|---|---|---|---|---|
| 1 — pilot | 34 (+4 scripted baselines) | **49,977** (32,403 in / 17,574 out) | — (budget-sizing phase) | n/a (arms pooled) |
| 2 — cross-case | 576 (289 cold / 287 warm) + 70 untokenized baselines | **912,003** (435,443 cold / 476,560 warm) | repeats held at the 4-per-arm floor | **+41,117 (+9.4%)** |
| 3 — compounding | 508 live (257/251) + 132 replay calls + 60 zero-token baseline replays | **1,037,672** (902,962 invocation / 134,710 promotion replay) | 79.8% of the 1.3M stop line; drift 519,146/520K, billing 231,399/390K, audit 287,127/390K | **+28,140 (+6.4%)** |
| **Program** | | **1,999,652** (~2.0M) | every per-case and per-phase stop line respected | 6.4–9.4% |

Accounting caveats, all disclosed in the phase reports:

- Phase-2 baseline records carry no token fields, so baseline spend is absent from the
  phase-2 totals.
- The phase-3 per-case `spend-*.json` snapshots are written at block boundaries and lag
  each case's final promotion pass — summing them undercounts (983,286). The
  authoritative total is consolidated-rollout tokens (902,962) plus the promotion
  ledger's replay count (134,710).
- This report re-verified the phase-3 arithmetic against the committed raw data: the
  ledger's per-pass replay in/out sums to 134,578; the phase-3 report's authoritative
  134,710 differs by 132 tokens, which sits in the final-pass reconciliation between
  the lagging snapshots and the per-case totals (drift +24, audit +108).
- Phase-3's committed tree carries 28 of 30 invocation reports — see §9.4.

## 7. Central scientific finding

**Warm-start retrieval is never worse than cold, with a strict noise-floor-clearing
advantage on drift-version-skew only — and that advantage persisted across generations
but did not compound.**

1. **Never worse.** Across all 67 paired both-arms invocations in the program, warm ≥
   cold at the case level everywhere. The only cold-favoring paired flips in the entire
   dataset are 1 of 16 partial-batch-write pairs (phase 2) and 4 of 48 drift pairs
   (phase 3) — all sub-floor, and no case-level result favors cold.
2. **Specific, not broad.** Exactly one case clears its noise floor:
   `drift-version-skew`, the case whose failure class (version-skew drift) is what the
   seeded learnings describe. Phase-3 pooled: warm 17/48 vs cold 6/48 accepted (~3×),
   median outcome 1.0 vs 0.0, net paired delta +11 above the 0.7413 floor.
3. **Persisted, not compounded.** The warm margin was flat across all three generation
   blocks (6/16 → 6/16 → 5/16 vs a constant 2/16) — no growth and no decay. Nine
   production promotion passes decided 11 proposals and promoted 0, leaving the store
   at its 12 seeds. With a static store, retrieval injected a constant payload (warm
   premium +9.4% → +6.4%), so phase 3 measures **retention, not growth**: v1's data
   rule out retrieval decay as the cause of the missing compounding and point at the
   gate (0/11 approvals). Additional blocks over a static store would re-measure
   retention at linear cost.
4. **What v1 does not show:** broad warm lift; growth-based compounding; behavior
   behind the Docker production boundary; any provider or model other than Anthropic
   `claude-sonnet-4-5`.

## 8. Threats to validity

- **Small case set.** 10 cases; effectively 2 discriminate anything (`drift-version-skew`
  strictly, `partial-batch-write` marginally); 5 saturate at the acceptance ceiling and
  2 sit at the floor. Phase 3 carries 3 of the 10. Single-case-wide conclusions are the
  program's real evidentiary width.
- **The structural promotion null.** 0/11 promotions cannot distinguish "no proposal
  was ever good enough" from "the gate cannot approve anything." The scoped regression
  guard never guarded anything, so its production behavior is unexercised at v1.
- **Single provider, single model.** Anthropic `claude-sonnet-4-5`, temperature 0.0,
  thinking off, throughout. No cross-provider or cross-model evidence.
- **Warm token premium 6.4–9.4%.** Retrieval injection is not free; the premium was
  constant only because the store never grew. A growing store's premium is untested.
- **Saturation and floor controls.** Saturated cases (both arms 16/16 or 12/12) and
  floor cases (both arms 0) cannot register retrieval effects; they bound harm, which
  held exactly.
- **Sample size and variance.** n=4 paired repeats per case per block; drift's MAD
  floor (0.7413) is carried by warm's own repeat variance (0–3 accepted per invocation).
  Repeats stayed at the directive's 4-paired-repeat floor absent a variance-driven
  reason to expand.
- **Executor boundary.** `executor: local` (LocalSubprocessSandbox) throughout — the
  trusted-subprocess executor, not the production Docker boundary; the C-level
  bytecode-verifier blind spot applies. All phase-2/3 live invocations ran under the
  `--allow-local-live` opt-in (PR #18 refuses that combination by default).
- **v1 gate semantics, disclosed and not fixed mid-run.** The binary-delta MAD
  materiality floor is vacuous (2-of-3 flips promote as material); the oracle-leak scan
  skipped suite-less cases until PR #17 (phases 2–3 ran under the fixed guard);
  infrastructure failures can be misclassified as candidate failures.

## 9. Incident disclosure (complete)

**9.1 Provider stall and the PR #29 bounded-ladder recovery (phase 2, Sept 17,
~22:00–22:44 UTC).** Three live workers stalled on single model requests with
established TLS connections and no bytes returned; Anthropic's status page reported
the API operational, so the stall was request-specific. 8 of 40 planned invocations
were lost or never started. The stall motivated PR #29 (explicit connect/read/write/
pool timeouts, max_retries 1) and PR #28's typed failed-candidate degradation; after
both merged, 3 of the 8 missing invocations were recovered under the bounded ladder
(concurrency-account-ledger r4, partial-batch-write r3 and r4).

**9.2 Sandbox pause/resume effects (phase 2, ~22:44–23:01 and ~23:44–23:59 UTC).**
The sandbox paused and resumed twice during the recovery window, killing in-flight TCP
calls and restamping process start times. Both recoveries ran through the bounded
ladder; one invocation (partial-batch-write r3) was killed mid-run and re-run cleanly
when its scratch was found to contain records from both a pre-pause and a post-resume
attempt. Wall-clock continuity for per-call latency is approximate across these
windows; per-call `latency_ms` values recorded inside each process are unaffected.

**9.3 Typed degraded candidate (phase 2, bounded machinery observed live).**
partial-batch-write r3's warm generation-0 builder consumed the full 600 s read
timeout plus its one bounded retry, then degraded to a recorded **typed failed
candidate** while the run continued; the invocation completed 18 records with 15/16
warm candidates accepted — the fail-closed path working as designed, disclosed rather
than hidden.

**9.4 Phase-4 provenance note (new in this report).** The committed phase-3 tree
carries **28 of 30** invocation reports: `audit-ring-drop-g3-r2/r3` and the audit g3
engine archive are absent from `runs/compounding/reports/` and `runs/compounding/
engine-archive/`, although the promotion ledger shows the audit g3 pass ran (3
decisions) and the spend ledger records 9 audit invocations done. This report
re-derived the pooled tallies from the committed raw data and they reconcile exactly:
committed reports tally drift 17/48 vs 6/48, billing 36/36, and audit 0/28 over the 7
committed audit invocations, with committed invocation tokens summing to 859,832 — the
43,130-token gap to the reported 902,962 equals precisely two audit-scale invocations
(~21.6K each, matching audit's per-invocation average (287,127 − 93,065) / 9 ≈ 21.6K).
The audit 0/36 pooled figure rests on the phase-3 report's full-run accounting for the
two uncommitted reports; the committed 0/28 is consistent with it.

**9.5 v1 gate-semantics caveats (carried across phases, never fixed mid-run).** Listed
in §8: the vacuous binary-delta MAD materiality floor, the suite-less oracle-leak gap
(pre-#17; phases 2–3 ran under the fixed guard), and infrastructure-failure
misclassification risk.

**9.6 Phase-1 live findings.** The fenced-JSON extraction defect, the
collapsed-candidate retention contract, and the bridge latency drop were found live
and fixed in-phase (§3); the pilot's known-gate-semantics list is the origin of the v1
caveats carried through the program.

**9.7 Phase-3 operational disclosures.** Zero degraded markers, no stalls, no rerun-
utility use: all 30 invocations and all nine passes completed sequentially, and the
complete dry run preceded any live call. Operator-level: the coordinating worker
exhausted its platform wake budget twice mid-run on fallback timers (phase 2's failure
mode was self-timer polling; phase 3 used a detached watchdog instead). The benchmark
was never touched — both tmux sessions ran uninterrupted, no invocation was restarted
or duplicated, and all evidence through block 6 was checkpoint-pushed before the
revival.

**Secrets hygiene, program-wide.** Pilot: 0/38 records tainted. Cross-case: 0/657
records tainted (9 canonical + 2 preflight files). Phase 3: gitleaks passed in CI at
every pushed head (including checkpoint tip `1efbf84`); no credential appears in any
configuration, finding, archive, or rollout record.

## 10. Checkpoint index (rollback points)

Every merged PR #1–#30, squash-merged to `main` (SHA = the rollback point):

| PR | SHA | Merged change |
|---|---|---|
| #1 | `c444182` | feat: scaffold evogine v1 |
| #2 | `740dde8` | feat(bench): decay-pressure task format, harness, 10-case corpus |
| #3 | `7f9b20d` | feat(engine): git-backed learning core with audit trail and seed bootstrap |
| #4 | `2ba6a57` | docs(seeds): record verified warm-start state of the seed pack |
| #5 | `267fa18` | feat(gates): port objective verification gates from Darwin |
| #6 | `10e387a` | feat(swarm): task loop with builder, verifier, reflector roles |
| #7 | `8e9ea75` | feat(gates): memory gate — counterfactual replay promotion for learnings |
| #8 | `b922ddf` | feat(swarm): wire the loop to the real code gate, harness, and CLI |
| #9 | `ec3b994` | **feat(bench): phase-1 pilot — paired warm/cold dataset, configs, and report** |
| #10 | `6ba41c9` | ci: close the Docker sandbox CI coverage gap for the gate tests |
| #11 | `938f378` | fix(swarm): point the oracle-leak guard at the live prompt path |
| #12 | `5907793` | fix(gates): reject candidate-staged file names that escape the staging root |
| #13 | `e92e498` | fix(gates): give the local executor a minimal allowlisted environment |
| #14 | `41d2383` | feat(swarm): wire the promotion loop into the production run path |
| #15 | `29146f6` | ci: add pinned gitleaks secret-scanning job to CI |
| #16 | `047c147` | fix(swarm): route temperature through extra_body for anthropic>=1.6 |
| #17 | `e212153` | fix(gates): run the oracle-leak scan on suite-less single-function cases |
| #18 | `1e53c8c` | feat(cli)!: refuse executor:local paired with a live provider |
| #19 | `5f8db1f` | chore(pkg): release-quality packaging metadata |
| #20 | `184ffb6` | fix(gates): fence reflector evidence and scan promotions for oracle leaks |
| #21 | `891e994` | chore: case-id containment, host lockfile, drop dead bench modules |
| #22 | `beef5c7` | test(promotion): drive a package-layout candidate through the full loop |
| #23 | `c061381` | test: close fail-closed validation gaps and pin an 88% coverage floor in CI |
| #24 | `acd8435` | **feat(bench): phase-2 cross-case warm-vs-cold benchmark with findings** |
| #25 | `7f93442` | test(gates): pin the fail-closed contract for verifier, replay, and sandbox faults |
| #26 | `976b8e5` | fix(gates): give LocalSubprocessSandbox DockerSandbox-style startup grace |
| #27 | `d93636a` | fix: make full-repo mypy clean and add the type-check to CI |
| #28 | `f63fbdb` | fix(swarm): degrade model-call faults and unify seed derivation |
| #29 | `74f707b` | fix(swarm): bound live provider requests with explicit timeouts |
| #30 | `6a2af31` | **feat(bench): phase-3 compounding benchmark — multi-generation warm persistence** |

Repo root before PR #1: `7717f65` (chore: initialize evogine repository).

## 11. What the evidence says v2 should change about promotion machinery

Grounded strictly in the phase reports — the full v2 roadmap is authored elsewhere.

1. **Study the gate before buying generations.** Phase 3's own verdict: the binding
   constraint is the promotion gate's approval rate (11 decisions, 0 promotions); more
   blocks over a static store re-measure retention at linear cost. The ledger records
   outcomes (`ephemeral`) but not *why* — decision-path instrumentation (rejection
   reasons per proposal) is the cheapest first step.
2. **Separate "nothing worth promoting" from "cannot promote."** The same 0/11 outcome
   covered a discriminator case with 17/48 warm acceptances (drift) and a floor case
   with 0/36 (audit): the gate's behavior was invariant to candidate-quality signal.
   v2 needs a diagnosis that distinguishes a correctly-strict gate from a stuck one —
   the 11 decided proposals are deterministically replayable from recorded runs, so
   threshold sweeps over them are feasible without new spend.
3. **Re-make the regression-guard scope decision once promotions exist.** The scoped
   guard (newly promoted records only) never guarded anything at 0 promotions, and the
   12 seeds were deliberately excluded to avoid replay cost (the memory gate's
   production default is three repeats). With real promotions, that exclusion needs
   re-validation against measured guard cost.
4. **Size replay against expected approval rate.** Promotion replay cost 134,710
   tokens (~13% of phase-3 spend) and produced zero promotions — per-pass replay is
   paid regardless of outcome, so v2 should budget it proportionally to expected
   approvals rather than per block.
5. **Grow the discriminating case set, not the repeat count.** One case in ten
   discriminates retrieval. Phase 2's verdict ("no evidence of broad warm advantage at
   this sample size") and the pilot's power analysis (the binary accepted-bit is too
   coarse; medians + MAD floors adopted instead) point the same way: v2's evidentiary
   value comes from more drift-like cases, while repeats hold at the 4-paired-repeat
   floor unless variance argues otherwise.
6. **Treat the warm premium as a defended budget line.** Phase 2 measured +9.4% and
   phase 3 +6.4% with a static store; a store that actually grows will move this
   number, and 6.4–9.4% is the v1 baseline any v2 retrieval change must justify
   against.

## Sources

- `reports/PILOT.md`, `reports/CROSSCASE.md`, `reports/COMPOUNDING.md` — the three
  merged phase reports; primary sources for every quoted number.
- `runs/pilot/` — `rollouts.jsonl` (clean dataset) plus preflight archives.
- `runs/crosscase/` — `results.csv` (per-case outcomes and per-run paired delta
  vectors), `crosscase.yaml`, `rollouts.preflight.unicode-nfkc-key-r4.jsonl`.
- `runs/compounding/` — `promotion-passes.jsonl` (9 passes, 11 decisions, 132 replay
  calls), `decided/` ledgers (9 audit + 2 drift proposals), `spend-*.json` (per-case
  stop lines), 28 invocation reports, 8 engine archives.
- `runs/compounding-dry/` — the zero-token scripted dry run (30 invocation reports,
  `analysis.json`, `results-pooled.csv`).
- Git history on `main` — PR #1–#30 merge SHAs (§10).
