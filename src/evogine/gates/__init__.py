"""Objective gates: deterministic, independent verdicts on what the swarm proposes.

The code gate (:mod:`evogine.gates.code_gate`) is the objective verifier for
candidate code changes: sandboxed pytest, differential equivalence at the
public API, and never-worse-than-baseline selection. The memory gate
(:mod:`evogine.gates.memory_gate`) promotes learned knowledge on
counterfactual replay: a learning is promoted only when the replay harness
(:mod:`evogine.gates.replay`) shows it would have flipped a past run's gate
verdict above the per-case materiality floor. The supporting primitives —
the locked-down sandbox, the deterministic bytecode-instruction verifier, and
the equivalence machinery — live in their own modules and are re-exported
here for gate consumers.
"""

from __future__ import annotations

from evogine.gates.code_gate import (
    OPS_DISCLOSURE,
    CodeGate,
    GateInput,
    GateResult,
    GateVerdict,
    GoldenCase,
    compare_to_baseline,
    find_oracle_leak,
)
from evogine.gates.equivalence import (
    ReplayCrash,
    ReplayError,
    ReplayTimeout,
    canon,
    canon_outcome,
    check_function_equivalence,
    check_unit_equivalence,
    load_unit,
    run_sequence,
)
from evogine.gates.memory_gate import GATE_ACTOR, MemoryGate, MemoryGateConfig
from evogine.gates.replay import (
    MAD_TO_FLOOR,
    AppliedPatch,
    LearningApplier,
    RecordedRun,
    ReplayEvidence,
    RunLog,
    RunReplayResult,
    UnknownRunError,
    flip_is_material,
    paired_delta_and_floor,
    regression_is_material,
    replay_one,
)
from evogine.gates.sandbox import (
    DockerSandbox,
    Executor,
    LocalSubprocessSandbox,
    RunResult,
    RunStatus,
    SandboxUnavailable,
)
from evogine.gates.verifier import (
    VerifierError,
    VerifierTimeout,
    measure_ops,
)

__all__ = [
    "GATE_ACTOR",
    "MAD_TO_FLOOR",
    "OPS_DISCLOSURE",
    "AppliedPatch",
    "CodeGate",
    "DockerSandbox",
    "Executor",
    "GateInput",
    "GateResult",
    "GateVerdict",
    "GoldenCase",
    "LearningApplier",
    "LocalSubprocessSandbox",
    "MemoryGate",
    "MemoryGateConfig",
    "RecordedRun",
    "ReplayCrash",
    "ReplayError",
    "ReplayEvidence",
    "ReplayTimeout",
    "RunLog",
    "RunReplayResult",
    "RunResult",
    "RunStatus",
    "SandboxUnavailable",
    "UnknownRunError",
    "VerifierError",
    "VerifierTimeout",
    "canon",
    "canon_outcome",
    "check_function_equivalence",
    "check_unit_equivalence",
    "compare_to_baseline",
    "find_oracle_leak",
    "flip_is_material",
    "load_unit",
    "measure_ops",
    "paired_delta_and_floor",
    "regression_is_material",
    "replay_one",
    "run_sequence",
]
