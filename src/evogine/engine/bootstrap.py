"""Bootstrap: load human-promoted seed records into the engine store.

The seed pack (``src/evogine/seeds/``) holds human-authored, human-promoted
learnings that give the engine a warm start. Seeds enter the store exactly
as authored — promoted by human decision, not by the memory gate — so the
loader refuses non-promoted seed files rather than silently re-deciding
their status. The whole pack commits as a single bootstrap mutation, with
one audit line per seed; a later ``Engine`` construction rehydrates the
promoted seeds into retrieval.
"""

from __future__ import annotations

from pathlib import Path

from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine


def load_seeds(engine: Engine, path: str | Path) -> list[LearningRecord]:
    """Load promoted seed records from ``path`` into ``engine``.

    Reads every ``*.json`` file in ``path`` (sorted by name), validates each
    against the pinned :class:`LearningRecord` schema, requires
    ``status == "promoted"``, and commits the pack to the store as one
    bootstrap load (one commit, one audit line per seed).

    Returns the loaded records. Targets a fresh store: a seed id already
    present raises ``ValueError`` — bootstrap is not idempotent by design,
    and a duplicate load is a caller bug the engine will not paper over.

    Raises:
        ValueError: a seed file is not a valid record, is not promoted, or
            its id already exists in the store.
    """
    seed_dir = Path(path)
    records: list[LearningRecord] = []
    for seed_file in sorted(seed_dir.glob("*.json")):
        try:
            record = LearningRecord.model_validate_json(seed_file.read_text(encoding="utf-8"))
        except Exception as error:
            raise ValueError(f"invalid seed file {seed_file}: {error}") from error
        if record.status != "promoted":
            raise ValueError(f"seed {seed_file} has status '{record.status}'; seeds must be promoted")
        records.append(record)
    if not records:
        return []
    return engine._ingest_batch(records)
