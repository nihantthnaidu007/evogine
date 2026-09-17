"""Rollout records: the benchmark's audit trail, in the phase0 schema.

Every rollout — the baseline evaluation and each generated candidate — is one
JSON line in ``runs/rollouts.jsonl`` carrying the phase0 fields: ``stage``,
``case_id``, ``run_id``, ``generation``, ``candidate_index``, ``model``,
``config``, ``prompt_messages``, ``extracted_code``, ``verdict``,
``failure_detail``, ``ops``, ``tokens``, and ``timestamp``.

The schema is enforced on write: a record missing a field is a bug, not a
gap. Rollout records stay host-side; failure details may name hidden
property ids and expected values, so they must never be rendered into a
builder prompt (tests/test_oracle_leak.py guards the prompt path).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROLLOUT_FIELDS = (
    "stage",
    "case_id",
    "run_id",
    "generation",
    "candidate_index",
    "model",
    "config",
    "prompt_messages",
    "extracted_code",
    "verdict",
    "failure_detail",
    "ops",
    "tokens",
    "timestamp",
)


def build_rollout(**fields: Any) -> dict[str, Any]:
    """Assemble one rollout record with the phase0 schema and a UTC timestamp.

    Keyword arguments must cover every field except ``timestamp``, which is
    stamped here; a missing field raises rather than silently thinning the
    audit trail.
    """
    missing = [name for name in ROLLOUT_FIELDS if name != "timestamp" and name not in fields]
    if missing:
        raise ValueError(f"rollout record missing fields: {missing}")
    record: dict[str, Any] = {name: fields[name] for name in ROLLOUT_FIELDS if name != "timestamp"}
    record["timestamp"] = datetime.now(UTC).isoformat()
    return record


class RolloutWriter:
    """Appends rollout records as JSON lines to a ``rollouts.jsonl`` file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        """Append one record; reject records missing any phase0 field."""
        missing = [name for name in ROLLOUT_FIELDS if name not in record]
        if missing:
            raise ValueError(f"rollout record missing fields: {missing}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
