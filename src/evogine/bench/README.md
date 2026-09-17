# The decay-pressure benchmark: task format and harness

Decay pressure is the gap between "tests green" and "survives production": a
baseline that looks fine on the happy path and fails under pressure — DST
boundaries, timezone math, concurrent writes, retry storms, partial failure,
load spikes, unicode/locale, overflow, integration drift, silent data
corruption. This benchmark extends the static degradation categories of the
Darwin corpus (see `nihanthnaidu007/Darwin`, `docs/authoring-a-case.md`) toward
that production-like decay, and the harness measures whether promoted engine
knowledge (warm) beats a cold start on the same model budget.

**Load-bearing premise:** the claim "the engine helps" is only measurable if
cases actually discriminate — the degraded baseline must pass every visible
test and fail hidden invariants under pressure. If a case's hidden properties
leak into prompts, or its visible suite fails under pressure, a warm-vs-cold
delta is meaningless. The oracle-leak scan (`tests/test_oracle_leak.py`) and
the case invariants (`tests/test_cases.py`) are the standing guards of that
premise.

## Case format

Each case lives in `src/evogine/bench/cases/<case_id>/`:

| File | Role | Generator sees it? |
|---|---|---|
| `reference.py` (or `reference/`) | The clean, correct target — host-side behavioral ground | no |
| `degraded.py` (or `degraded/`) | The plausible v1 that decays; the builder's starting point | yes (as the repair input) |
| `test_suite.py` | Visible happy-path tests the degraded version PASSES — pressure exposes what they miss | sandbox-side (candidate gate) |
| `properties.json` | Hidden invariants checked host-side by the gate; **never in any prompt** | never |
| `meta.json` | `case_id`, `decay_axis`, `difficulty`, `description` | yes |
| `scenario.md` | What pressure breaks it — oracle-free prose | yes |

Two layouts: **single** (one-module contract; `reference.py`/`degraded.py`) and
**package** (multi-file port-style units in the manner of Darwin's `mcases/`;
`reference/` and `degraded/` package directories). Candidate code is always
materialized under the module name `candidate` (single: `candidate.py`;
package: `candidate/<module>.py`), and the visible suite imports from it:

```python
from candidate import billing_day            # single
from candidate.queue import JobQueue         # package
```

## Authoring discipline

1. **Reference first, by rubric.** Deterministic, stdlib-only, no I/O, no
   wall clock, no randomness, no threads. Pressure is *modeled* with explicit
   deterministic inputs: a modeled failure set, a logical clock, a hash-based
   jitter, a large-but-finite sequence. The constraint string on every case
   is the Darwin one: `deterministic, no I/O, no wall clock, no randomness`.
2. **JSON-serializable contracts.** Entry points return values that survive
   a JSON round trip (strings, ints, bools, lists, dicts of those); freeze
   and replay compare structurally. Ints over floats — monetary and counter
   math stays in exact integers so replay is bit-exact.
3. **Choose pressure inputs, then freeze expected values from the reference**
   with `python tools/freeze_properties.py --case <case_id>`. The frozen
   values are the reference's recorded behavior on hand-chosen inputs — the
   Darwin golden-I/O discipline. If the reference later changes, refreezing
   fails loudly where behavior changed.
4. **Degraded is plausible, not broken.** It must pass the visible suite as
   written and fail at least one hidden property. Write the visible suite
   like a real v1 team: happy-path coverage that a careful reviewer would
   accept, with the pressure cases simply not present.
5. **properties.json is data, not code.** Two generic kinds, checked by
   `bench/gate.py` host-side:
   - `golden_io` — `{"args", "kwargs", "expected"}` or `{"args", "kwargs",
     "raises"}` rows against the case entry symbol.
   - `scenario` — `{"construct", "steps"}` rows replaying a deterministic
     method sequence on a fresh object (how stateful pressure is pinned
     without threads); optional `construct_raises`.
6. **No oracle in generator-visible text.** `scenario.md` and the prompt
   templates in `bench/prompts.py` must never contain an expected value from
   any properties.json. The scan is union-wide across the corpus.

## Harness

`bench/harness.py` exposes `run(task, engine, config, gate=None)`. Each
selected arm delegates to `evogine.swarm.runner.run_task` — the real loop end
to end. Injection points are explicit: retrieval (the warm arm takes the
`engine`; the cold arm retrieves via `NullEngine`), generation
(`config.builder`, a benchmark `ModelClient` — CI uses scripted models, no
API calls), and evaluation (`gate`, defaulting to `CodeGateAdapter`, which
routes every candidate through the real sandboxed code gate). Retention is
best-of-N with a never-worse fallback to the degraded baseline, decided
inside the loop. The returned report pairs the arms: `arms` maps each arm
label to its loop report (tagged with `arm`), with the first arm's baseline,
`rollouts_path`, and run identity surfaced at the top level. Every rollout —
baseline and candidates — lands in `runs/rollouts.jsonl` in the phase0 schema (`stage`,
`case_id`, `run_id`, `generation`, `candidate_index`, `model`, `config`,
`prompt_messages`, `extracted_code`, `verdict`, `failure_detail`, `ops`,
`tokens`, `timestamp`).

Determinism defaults follow the learn_0012 seed playbook: thinking off,
temperature pinned at 0.0, and a per-record seed
`sha256(run_id|case_id|generation|candidate_index)` derived and recorded on
every rollout. Arms of one run share `run_id`, so cold and warm candidates at
the same `(generation, candidate_index)` are seed-paired and retrieval is the
only difference between the arms.

## Adding a case

1. Create `bench/cases/<case_id>/` with `meta.json`, `scenario.md`,
   `reference.py`, `test_suite.py`, and a `degraded.py`.
2. Add a `properties.json` skeleton with `case_id`, `entry`, and property
   groups whose inputs expose the pressure; run
   `python tools/freeze_properties.py --case <case_id>` to freeze expected
   values from the reference.
3. Verify the three invariants locally: reference passes all properties,
   degraded passes the visible suite, degraded fails at least one property —
   `pytest tests/test_cases.py -q` covers every case in the corpus.
4. Confirm the oracle-leak scan passes: `pytest tests/test_oracle_leak.py -q`.
