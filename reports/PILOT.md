# Phase 1 Pilot — Warm vs Cold, End to End

Status: **complete**. Pin: `019866a` (feat/bench-pilot). Scope: 2 cases × both arms
(warm engine retrieval, cold NullEngine), 1 paired run per case (both arms share one
`run_id`, so `(generation, candidate_index)` seeds match across arms and only retrieval
differs), `candidates_per_generation=4`, temperature 0.0, thinking off, per-record seeds =
sha256(run_id|case_id|generation|candidate_index)[:8], recorded in every rollout.

- Case 1 `audit-ring-drop` — single-function golden_io scope.
- Case 2 `concurrency-account-ledger` — package scenario scope.

## 1. What validated / what did not

**Validated end to end with a real model:** harness → swarm runner (builder → verifier →
gate) → sandboxed code gate (local executor) → retention → reflector → rollout records.
Rollout completeness: 38/38 records carry the full phase0 schema plus the learn_0012
determinism block (32/32 candidate+verifier records checked, 0 missing). Warm retrieval
returned promoted seed records; cold proposals were rejected without a backing store.

**What did not validate (found live):**
1. *Fenced JSON package replies broke extraction* — the real defect, fixed in this PR.
   The model fences its `{"files": ...}` reply as ```` ```json ```` even when the
   contract asks for the raw object (8/8 live replies fenced; the scripted CI provider
   masks this), and `extract_code`'s fence regex only knew the `python` tag, so the
   `json` tag leaked into the payload and every package candidate failed extraction.
   Fixed (tag-agnostic regex + defensive fence strip) with regression tests.
2. *Collapsed-candidate retention is a deliberate contract, not a bug.* The pilot
   initially read the vacuous 0/0 counters (an ungated candidate retained as `best`,
   reflector silent) as a retention defect and changed the eligibility rule — then the
   #14 contract test on merged main
   (`test_collapsed_candidate_records_no_replay_run`) showed that semantics are
   intentional: a candidate with no re-evaluable gate evidence must not reach the
   reflector, so no learning is fabricated from nothing. This PR restores the #14
   behavior unchanged. Design observation carried to the findings report: the reflector
   therefore never diagnoses builder-level (extraction) failures; with the fence fix the
   trigger is gone in practice, but the blind spot is structural.
3. *Bridge latency was dropped* converting bench↔swarm response protocols (fixed earlier
   on this branch, commit `c538b4f`).

## Token spend (budget sizing for phases 2–4)

Across all 38 records (2 cases × 2 arms, one generation, 4 candidates each):

| stage | calls | input tokens | output tokens |
|---|---|---|---|
| baseline | 4 | 0 (scripted) | 0 |
| candidate (builder) | 16 | 13,608 | 6,170 |
| verifier | 16 | 17,521 | 10,503 |
| reflector | 2 | 1,274 | 901 |
| **total** | **34** | **32,403** | **17,574** |

Per case (both arms, one paired invocation):
- Case 1: builder 5,896/1,640 · verifier 6,856/4,830 · reflector 1,274/901 → ~13.9K in, ~7.4K out.
- Case 2: builder 7,712/4,530 · verifier 10,665/5,673 · reflector 0 (no failures) → ~15.4K in, ~10.2K out.

Per-arm per-generation cost (phases 2–4 sizing unit) ≈ **16.2K input + 8.8K output tokens**
for a 4-candidate, 1-generation case run, minus reflector when everything passes.
Phase-2 projection (10 cases × 2 arms × 4 repeats): ≈ 650K input + 350K output tokens,
assuming pilot-like case difficulty. **Phase 2 stays at the 4-paired-repeat floor** (directive 3).

## Latency

| stage | case 1 (8+2 calls) | case 2 (16 calls) |
|---|---|---|
| builder | 35.3 s | 57.5 s |
| verifier | 121.6 s | 115.2 s |
| reflector | 30.1 s | 0 s (no failures) |

Wall-clock per case (both arms, one invocation): ≈ 3 min. Verifiers dominate (~60% of
latency, ~54% of output tokens).

## Variance + power check

Paired warm−cold accepted-bit deltas: case 1 `[0, 0, 0, 0]`, case 2 `[0, 0, 0, 0]`.
All 8 paired deltas are exactly zero — the pilot's binary outcome has **zero variance**
under the paired design. learn_0009 noise floor: 1.4826 × MAD of centered repeats = 0 for
degenerate (all-identical) repeats; every delta is far under any floor and is labeled
**indistinguishable from noise / structural tie**, per case and pooled.

Power check: with zero observed variance in binary outcomes, no repeat count resolves a
nonzero delta better than the pilot did — the binary accepted-bit is too coarse at this
difficulty. Phase 2 therefore keeps the **4-paired-repeat floor** (directive: do not
multiply repeats without a variance-driven reason) and reports per-case medians + MAD
floors over run-level outcome scores, with all-zero-delta cases labeled
indistinguishable from noise.

## Pilot result (honest headline)

- **Case 1 `audit-ring-drop`: 0/8 accepted, all 8 learn_0011 never-worse rejections
  (measured structural tie).** The degraded baseline measures 44 bytecode ops; all eight
  repairs also measure 44; retention requires strict ops improvement. The gate is doing
  its job; the case cannot discriminate retrieval benefit. Disclosed, not tuned.
- **Case 2: 8/8 accepted — 4/4 in BOTH arms**, paired deltas zero. The repair is solvable
  with and without retrieval at this difficulty. Retrieval benefit is not demonstrated
  at pilot scale; that is the finding the cross-case benchmark is sized to test.
- Warm retrieval fired correctly (warm arm retrieved seeded records; cold retrieved none)
  with no prompt contamination.

## Known gate semantics disclosed (v1, not fixed mid-run)

- Oracle-leak scan: at pilot pin, ran only for single-function scope with a test suite
  (`code_gate.py:315` guard); suite-less cases skipped it. PR #17 (suite-less scanning)
  merged to main after this run — phases 2–3 run under the fixed guard.
- Binary-delta MAD materiality floor is vacuous (2-of-3 flips promote), unchanged at 3 repeats.
- Infrastructure failures can be misclassified as candidate failures.
- No Docker daemon in this environment — `executor: local` (LocalSubprocessSandbox),
  the trusted-subprocess executor, not the production boundary. Accepted residual,
  re-disclosed in the final findings.

## Secrets hygiene

All 38 rollout records scanned (`failure_detail`, `extracted_code`, `reasoning_content`)
for ANTHROPIC key material (`sk-ant-` / `antrop` case-insensitive): **0 tainted records**.
The run predated the #13 env-allowlist hotfix (full os.environ inheritance was open
during the pilot); no key material leaked into any record.

## Provenance

- Pre-fix attempts preserved: `runs/pilot/rollouts.preflight.jsonl` (orphaned baseline),
  `rollouts.preflight.d66921d.jsonl` (retention-fix pin, tag-leak failures),
  `rollouts.preflight.case2-unpaired.jsonl` (unpaired run_ids, superseded).
- Clean dataset: `runs/pilot/rollouts.jsonl` — case 1 from pin `c538b4f` (records valid;
  the fence defect only affects package layouts), case 2 from pin `019866a`.
- Analyzer: `tools/analyze_rollouts.py`.
