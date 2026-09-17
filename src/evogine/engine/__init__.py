"""Engine: the durable, git-backed learning core of evogine."""

from __future__ import annotations

from evogine.engine.audit import AuditEntry, AuditLog
from evogine.engine.bootstrap import load_seeds
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.engine.store import Engine, InvalidTransitionError, RecordNotFoundError

__all__ = [
    "AuditEntry",
    "AuditLog",
    "Engine",
    "InvalidTransitionError",
    "LearningRecord",
    "RecordBody",
    "RecordCounterfactual",
    "RecordNotFoundError",
    "RecordProvenance",
    "load_seeds",
]
