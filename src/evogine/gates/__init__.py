"""Objective gates: deterministic, independent verdicts on what the swarm proposes.

The code gate (:mod:`evogine.gates.code_gate`) is the objective verifier for
candidate code changes: sandboxed pytest, differential equivalence at the
public API, and never-worse-than-baseline selection. The memory gate
(:mod:`evogine.gates.memory_gate`) promotes learned knowledge on
counterfactual replay (own task, still a stub). The supporting primitives —
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
from evogine.gates.memory_gate import MemoryGate
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
    "OPS_DISCLOSURE",
    "CodeGate",
    "DockerSandbox",
    "Executor",
    "GateInput",
    "GateResult",
    "GateVerdict",
    "GoldenCase",
    "LocalSubprocessSandbox",
    "MemoryGate",
    "ReplayCrash",
    "ReplayError",
    "ReplayTimeout",
    "RunResult",
    "RunStatus",
    "SandboxUnavailable",
    "VerifierError",
    "VerifierTimeout",
    "canon",
    "canon_outcome",
    "check_function_equivalence",
    "check_unit_equivalence",
    "compare_to_baseline",
    "find_oracle_leak",
    "load_unit",
    "measure_ops",
    "run_sequence",
]
