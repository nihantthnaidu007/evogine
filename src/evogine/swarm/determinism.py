"""Per-record seed derivation for the learn_0012 determinism playbook.

This is a leaf module on purpose (the way :mod:`evogine.swarm.fencing` is):
:mod:`evogine.swarm.runner` derives and records the seed on every rollout,
and :mod:`evogine.bench.harness` re-exports the same derivation for
benchmark-report callers, so it must not import either back.

The digest is truncated to 8 hex characters — the length every rollout
record has carried since the phase-1 pilot (``runs/pilot/rollouts.jsonl``
seeds are all 8 characters) and the length the loop's tests pin. Historical
run data stays interpretable against exactly this format; there is no other
seed-id format recorded anywhere.
"""

from __future__ import annotations

import hashlib


def derive_seed(run_id: str, case_id: str, generation: int, candidate_index: int) -> str:
    """Derive the per-record seed (learn_0012): sha256 of the record identity.

    Arms of one run share ``run_id``, so candidates at the same
    ``(generation, candidate_index)`` are seed-paired across arms — the
    comparison isolates retrieval as the only difference.
    """
    payload = f"{run_id}|{case_id}|{generation}|{candidate_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
