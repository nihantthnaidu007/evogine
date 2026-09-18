# Phase 2 — Cross-Case Benchmark Findings

Pinned provenance: evogine main `74f707b` (PR #29 client timeouts; PR #28 fail-closed
degradation; PR #14 promotion wiring; oracle-guard, staging-escape, env-allowlist
hotfixes). Branch `feat/bench-crosscase`. Config `runs/crosscase/crosscase.yaml`:
10 cases x (warm, cold) arms x 4 paired repeats, temperature 0.0, thinking off,
retrieval_k 5, 4 candidates/generation, 1 generation, executor local, provider
Anthropic `claude-sonnet-4-5`, learn_0012 per-record seeds.

## Headline

**Warm retrieval never loses to cold, ties 7 of 9 measured cases, and strictly wins
on 1: `drift-version-skew` (warm median 1.5 vs cold 0.5 accepted-outcome, delta +1.0
above the 1.4826 x MAD noise floor of 0.7413; warm accepted 6/16 candidates vs cold
2/16).** Two cases (`audit-ring-drop`, `unicode-nfkc-key`) are floor cases where both
arms accept zero candidates — no retrieval effect is possible there. The remaining
saturated cases (both arms 4/4) cannot discriminate. Warm costs +9.4% tokens over
cold (476,560 vs 435,443), the price of injecting retrieved learnings into prompts.

Per learn_0009 discipline: medians with per-case 1.4826 x MAD floors; deltas under
the floor are labeled indistinguishable from noise. Only drift-version-skew's paired
delta clears its floor.

## Per-case outcomes (medians, MAD floors, learn_0011 rejections)

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

Machine-readable table: `runs/crosscase/results.csv` (includes per-run paired
warm-minus-cold delta vectors). `counter-overflow-wrap` is a structural null — see
below.

## The one discriminating case

`drift-version-skew` is the only case whose outcome distribution separates warm from
cold: warm's candidates passed the objective gate on 6 of 16 attempts against cold's
2 of 16, with per-run paired flips concentrated in the warm arm and a median delta
(+1.0) that exceeds the 0.7413 MAD floor. This is directionally coherent with the
engine thesis — version-skew is exactly the failure class where remembering how a
prior version broke should help — but n=4 paired repeats and a 0/16 floor on two
other cases mean the honest summary is: **a positive, noise-floor-clearing signal on
one of nine cases, no evidence of harm anywhere, and no evidence of broad warm
advantage at this sample size.**

## Determinism (learn_0012) coverage

560 candidate/verifier records verified against the canonical seed derivation
`sha256(run_id|case_id|generation|candidate_index)[:8]`: **0 missing playbook blocks,
0 seed mismatches.** Baseline records (70) do not carry the candidate-index seed and
are excluded from that check by design.

## Token spend and latency per arm

| Arm | Model calls | Input tokens | Output tokens | Total | Wall latency (summed per-call) |
|---|---|---|---|---|---|
| cold | 289 | 272,987 | 162,456 | 435,443 | 3,186.8 s |
| warm | 287 | 305,929 | 170,631 | 476,560 | 3,387.9 s |

Combined 912,003 tokens across 35 completed invocations, ~26,000 per invocation —
matching the pilot's per-invocation projection, so repeats were held at the
4-per-arm floor per the budget directive rather than expanded. Warm's +41,117-token
(+9.4%) premium is the retrieval-injection overhead. Stage split (pooled):
candidates 280 calls (245,194 in / 116,941 out), verifiers 280 (320,840 in /
209,739 out), reflector 16 (12,882 in / 6,407 out). Known reporting gap: the 70
baseline records carry no token fields, so baseline spend is not represented in
these totals.

## Structural nulls, degraded evidence, and the API incident

Honest accounting of everything that did not go to plan, all preserved rather than
silently dropped:

1. **Provider stall (Sept 17, ~22:00-22:44 UTC).** Three live workers stalled on
   single model requests with established TLS connections and no bytes returned;
   the in-flight calls never completed. Anthropic's status page reported the API
   operational, so the stall was request-specific. Eight of 40 planned invocations
   were lost or never started.
2. **PR #29 fix + recovery.** The stall motivated PR #29 (explicit connect/read/
   write/pool timeouts, max_retries 1) and PR #28's typed failed-candidate
   degradation. After both merged, 3 of the 8 missing invocations were recovered
   under the bounded ladder (concurrency-account-ledger r4, partial-batch-write r3
   and r4).
3. **Degraded candidate (bounded machinery observed live).** partial-batch-write
   r3's warm generation-0 builder consumed the full 600 s read timeout plus its one
   bounded retry, then degraded to a recorded typed failed candidate while the run
   continued; the invocation completed 18 records with 15/16 warm candidates
   accepted. This is the fail-closed path working as designed, disclosed rather
   than hidden.
4. **Sandbox pause/resume incidents.** The sandbox paused and resumed twice during
   the resume window (~22:44-23:01 and ~23:44-23:59 UTC), killing in-flight TCP
   calls and restamping process start times. Both recoveries ran through the
   bounded ladder; one invocation (partial-batch-write r3) was killed mid-run and
   re-run cleanly when its scratch was found to contain records from both a
   pre-pause and a post-resume attempt. Wall-clock continuity for per-call latency
   is approximate across these windows; per-call latency_ms values recorded inside
   each process are unaffected.
5. **counter-overflow-wrap: 0 of 4 repeats.** The case's worker shard never started
   before the stall; per the coordinator's finality directive the phase is closed
   with this case as a structural null. Its four planned repeats are documented as
   missing, not imputed.
6. **unicode-nfkc-key: 3 of 4 repeats.** The r4 invocation was in flight when the
   queue was stopped for finality; its partial records are archived as
   `runs/crosscase/rollouts.preflight.unicode-nfkc-key-r4.jsonl` and excluded from
   analysis. The canonical dataset carries its 3 completed repeats (12
   candidates/arm).
7. **Executor.** Local subprocess sandbox throughout (no Docker in the environment);
   the C-level bytecode-verifier blind spot applies as documented in the pilot.
8. **Known v1 gate semantics** (carried from the audit and verification layers, not
   fixed mid-run): the binary-delta MAD materiality floor is vacuous (2-of-3 flips
   promote as material); the oracle-leak scan skips suite-less cases; infrastructure
   failures can be misclassified as candidate failures.

## Secrets hygiene

Scanned all 657 records (9 canonical + 2 preflight files) for Anthropic key
material (`antrop*` / `sk-ant-`) across failure_detail, extracted_code,
reasoning_content, and prompt_messages: **0 tainted records.** The env-allowlist
hotfix (PR #13) was in effect for the resumed invocations; phases 1-2 predate it —
the zero-taint result covers both.

## Verdict

The engine's retrieval advantage is real but narrow at n=4: warm >= cold on every
case, strictly better beyond the noise floor on exactly the one case whose failure
class matches what the seeded learnings describe. The compounding claim (phase 3)
tests whether that advantage grows across generations; this phase alone supports
"no harm, targeted help" — not "broad lift."
