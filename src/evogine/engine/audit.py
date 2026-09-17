"""Append-only audit log: every engine mutation is recorded, never rewritten.

The audit log is the engine's answer to "why do you believe that?" — every
propose, replay queueing, promotion, rejected promotion, demotion, and
bootstrap load appends one immutable line recording the timestamp, actor,
action, learning id, and the reason or evidence behind the decision. Lines
are never edited or removed; rejected knowledge stays visible so the engine
can learn from what it refused to learn.

The log is a JSONL file (one :class:`AuditEntry` per line) stored inside the
engine's git-backed data directory, so audit history is versioned by the
same commits that mutate the store.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError


class AuditEntry(BaseModel):
    """One immutable audit line: who did what, to which record, and why."""

    timestamp: datetime
    actor: str
    action: str
    learning_id: str | None = None
    reason: str | None = None
    evidence: str | None = None


class AuditLog:
    """Append-only JSONL audit log of every engine mutation.

    ``append`` is the only write path: it adds one line and never rewrites
    existing content. ``entries`` re-reads the log and refuses to skip
    corrupt lines — a damaged audit trail must surface, not silently shrink.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        action: str,
        *,
        actor: str = "engine",
        learning_id: str | None = None,
        reason: str | None = None,
        evidence: str | None = None,
    ) -> AuditEntry:
        """Append one audit entry for ``action`` and return it."""
        entry = AuditEntry(
            timestamp=datetime.now(UTC),
            actor=actor,
            action=action,
            learning_id=learning_id,
            reason=reason,
            evidence=evidence,
        )
        with self.path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry.model_dump(mode="json"), separators=(",", ":")) + "\n")
        return entry

    def entries(self) -> list[AuditEntry]:
        """Return every audit entry in append order.

        Raises:
            RuntimeError: if any line is not a valid :class:`AuditEntry`.
        """
        if not self.path.exists():
            return []
        parsed: list[AuditEntry] = []
        with self.path.open("r", encoding="utf-8") as log:
            for line_number, line in enumerate(log, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed.append(AuditEntry.model_validate_json(stripped))
                except ValidationError as error:
                    message = f"corrupt audit log at {self.path}, line {line_number}: {error}"
                    raise RuntimeError(message) from error
        return parsed

    def find(self, action: str, learning_id: str | None = None) -> list[AuditEntry]:
        """Return entries matching ``action`` (and optionally ``learning_id``)."""
        return [
            entry
            for entry in self.entries()
            if entry.action == action and learning_id in (None, entry.learning_id)
        ]
