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
├── engine/
│   ├── records.py     # LearningRecord: the versioned knowledge atom (pydantic contract)
│   ├── store.py       # Engine: git-backed lifecycle, promotion-only retrieval
│   ├── audit.py       # AuditLog: append-only audit trail of every engine mutation
│   └── bootstrap.py   # load_seeds: warm-start load of promoted seed records
├── gates/
│   ├── code_gate.py   # gate 1: sandboxed pytest + differential equivalence + never-worse
│   └── memory_gate.py # gate 2: counterfactual replay promotion of learned knowledge
├── swarm/
│   └── roles.py       # builder / verifier / reflector responsibilities
├── bench/
│   ├── harness.py     # warm-vs-cold decay-pressure benchmark
│   └── cases/         # benchmark cases (empty for now)
├── seeds/             # bootstrap learning records: 12 human-promoted seeds + provenance
└── py.typed
```

## Status

v1 engine core implemented: the git-backed `Engine` (propose / queue-replay /
promote / demote / retrieve, one commit per mutation), an append-only audit
log, and the bootstrap loader for the 12 human-promoted seed records. Gates,
swarm runner, CLI subcommands, and the benchmark remain staged for later
phases.

## Spec

Full v1 design: Blueprint artifact `art_bGBG4tqU` — "Self-evolving engine v1 — design
spec" (Obvious project `prj_knPCZEvt`).
