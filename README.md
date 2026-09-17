# evogine

evogine is a self-evolving agentic engine: a git-backed knowledge engine that detects
recurring failure patterns, preserves effective engineering practices, and improves code
repair through repeated, auditable, evidence-based evaluation. A task-scaled
builder / verifier / reflector swarm proposes changes and learnings; two objective gates —
a sandboxed **code gate** (sandboxed pytest + differential equivalence + never-worse check)
and a **memory gate** (counterfactual-replay promotion) — decide what lands and what is
remembered; a decay-pressure benchmark measures, warm versus cold, whether accumulated
knowledge actually improves outcomes.

## Layout

```
src/evogine/
├── cli.py             # evogine run / case list / case inspect (see "CLI" below)
├── engine/
│   ├── records.py     # LearningRecord: the versioned knowledge atom (pydantic contract)
│   ├── store.py       # Engine: git-backed lifecycle, promotion-only retrieval
│   ├── audit.py       # AuditLog: append-only audit trail of every engine mutation
│   └── bootstrap.py   # load_seeds: warm-start load of promoted seed records
├── gates/
│   ├── code_gate.py   # gate 1: sandboxed pytest + differential equivalence + never-worse
│   ├── equivalence.py # differential equivalence at the public API (replay oracle)
│   ├── verifier.py    # deterministic bytecode-ops measurement (never-worse bar)
│   ├── sandbox.py     # executors: Docker image or trusted local subprocess
│   ├── memory_gate.py # gate 2: counterfactual replay promotion of learned knowledge
│   └── replay.py      # counterfactual replay engine behind gate 2
├── swarm/
│   ├── roles.py       # builder / verifier / reflector clients
│   ├── runner.py      # the task loop: retrieve -> candidates -> gate -> retain -> reflect
│   ├── gates_adapter.py  # CandidateGate adapter wiring the loop to gates.CodeGate
│   └── rollouts.py    # enforced phase0 rollout records (JSONL audit per run)
├── bench/
│   ├── harness.py     # warm-vs-cold decay-pressure benchmark (see bench/README.md)
│   ├── task.py        # benchmark task loading and the case-corpus contract
│   └── cases/         # the 10-case decay-pressure corpus
├── seeds/             # bootstrap learning records: 12 human-promoted seeds + provenance
└── py.typed
```

## Status

v1 engine core implemented: the git-backed `Engine` (propose / queue-replay /
promote / demote / retrieve, one commit per mutation), an append-only audit
log, and the bootstrap loader for the 12 human-promoted seed records. The
swarm task loop is implemented: builder / verifier / reflector role clients,
warm retrieval into builder prompts, best-of-N retention with a never-worse
fallback to the baseline, and reflector learnings landing `ephemeral` through
`Engine.propose` — every step recorded in the phase0 rollout schema with the
learn_0012 determinism playbook (thinking off, pinned temperature, per-record
seeds). The loop is wired to the objective gates: `CodeGateAdapter` routes
every candidate through the sandboxed code gate (pytest + differential
equivalence + never-worse), the benchmark harness runs the real loop per arm
with retrieval injection (warm engine vs cold `NullEngine`), and the CLI
executes cases end to end, writing arm-labeled rollouts under the same
phase0 schema.

## CLI

The `evogine` entry point drives the wired loop: `run` executes one benchmark
case through retrieval -> builder -> the real code gate -> retention ->
reflector and writes phase0 rollout records; `case list` and `case inspect`
explore the corpus.

```console
$ evogine case list                     # case ids, one per line
$ evogine case inspect <case_id>        # decay axis, entry point, properties
$ evogine run --case <case_id> [--arm warm|cold] [--config <path>]
```

`run` prints one JSON report per selected arm (both arms by default). Without
a config it uses the deterministic scripted model and the Docker executor
(build it once: `docker build -f docker/Dockerfile.sandbox -t
evogine-sandbox:py313 docker/`). A YAML `--config` file may select
`executor: local` (trusted subprocess, no Docker), `model.provider:
scripted|anthropic`, `runs_dir`, `engine_root`, and the determinism knobs —
see the `evogine.cli` module docstring for the full key list.

Worked example, cold arm with the local executor (`local.yaml`:
`executor: local`):

```console
$ evogine case list
audit-ring-drop
billing-day-offset
...

$ evogine case inspect counter-overflow-wrap
case: counter-overflow-wrap
decay_axis: overflow
difficulty: medium
layout: single
entry: IdSource
suite: .../bench/cases/counter-overflow-wrap/test_suite.py
baseline files: candidate.py
description: An epoch-tagged id source with a bounded counter must saturate and report exhaustion, not wrap and reuse ids.
property: scenario id=exhaustion-reports-no-ids rows=2

$ evogine run --case counter-overflow-wrap --arm cold --config local.yaml
{
  "run_id": "c97c79f8d757",
  "case_id": "counter-overflow-wrap",
  "decay_axis": "overflow",
  "baseline": {
    "verdict": "fail_equivalence",
    "visible_failures": 1,
    "properties_passed": 0,
    "properties_total": 1
  },
  "arms": {
    "cold": {
      "retrieved_records": 0,
      "candidates": [
        {"generation": 0, "candidate_index": 0,
         "verdict": "fail_equivalence", "verifier_verdict": "fail", ...}
      ],
      "retained": "baseline",
      ...
    }
  },
  "rollouts_path": "runs/rollouts.jsonl"
}
```

The baseline and every candidate flow through the real gate: the degraded
incumbent fails differential equivalence, so the loop keeps it as the
incumbent rather than retaining a rejected repair. Every step — baseline
evaluation, builder calls, gate verdicts, retention, reflector drafts — is
recorded in `rollouts.jsonl` under the phase0 schema. With a seeded engine
and `--arm warm`, the warm arm injects promoted knowledge into the builder
prompt; arms of one run share `run_id`, so cold and warm candidates at the
same `(generation, candidate_index)` are seed-paired.

## Spec

Full v1 design: Blueprint artifact `art_bGBG4tqU` — "Self-evolving engine v1 — design
spec" (Obvious project `prj_knPCZEvt`).
