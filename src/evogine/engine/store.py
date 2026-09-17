"""Git-backed learning store: the engine's durable memory surface.

The :class:`Engine` owns the learning lifecycle (``ephemeral -> candidate ->
promoted`` or ``dropped``), versioned storage in git, and promotion-only
retrieval: swarm workers may retrieve only ``promoted`` records, keeping
unproven, agent-generated knowledge out of active use until the memory gate
proves it materially changes verdicts.

Storage model: plain JSON record files plus an append-only JSONL audit log
under a configurable data directory (default ``./data/engine``), versioned by
a dedicated git repository managed through subprocess ``git`` (no GitPython).
Every mutation — propose, queue-replay, promote, rejected promotion, demote,
bootstrap load — produces exactly one commit whose message names the mutation
and the learning id.

Retrieval relevance (v1, tracked open question): relevance is type/tag
matching against the task context — a record scores a hit when its ``type``
(spelled ``failure-pattern`` or ``failure pattern``) or one of its title
tokens (length >= 3) appears in the context. Tag-matching records rank
first; confidence orders records within a relevance tier; ties break by id
for determinism. When no promoted record matches, the top-``k`` promoted
records by confidence are returned anyway, so a warm engine never starves a
task of its knowledge. This is deliberately shallow: embedding-based
similarity is the candidate replacement, and the choice is reversible (see
blueprint ``art_bGBG4tqU``, open decisions).

Materiality (v1): ``promote`` requires replay evidence whose
``verdict_change`` is ``True`` backed by at least ``materiality_threshold``
replayed runs — the engine-level minimum. The memory gate
(``gates.memory_gate``) owns per-case floors on top of this minimum.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from evogine.engine.audit import AuditLog
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)

#: A learning id doubles as a record filename — keep it to a safe subset.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_PROMOTED = "promoted"
_CANDIDATE = "candidate"
_EPHEMERAL = "ephemeral"
_DROPPED = "dropped"


class RecordNotFoundError(LookupError):
    """No record with the requested learning id exists in the store."""


class InvalidTransitionError(ValueError):
    """The requested lifecycle transition is invalid for the record's status."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Engine:
    """Durable, git-backed store of :class:`LearningRecord` objects.

    Lifecycle contract (v1 design):
    ``propose`` stores a draft as ``ephemeral``; ``queue_replay`` attaches a
    pending counterfactual replay and marks the record ``candidate``;
    ``promote`` marks a record ``promoted`` (with ``promoted_at`` stamped)
    only when the replay evidence is material; ``demote`` moves a promoted
    record to ``dropped``. Rejected knowledge stays in the store, auditable,
    never deleted.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data/engine",
        materiality_threshold: int = 1,
        actor: str = "engine",
    ) -> None:
        """Open (creating if needed) the git-backed store at ``data_dir``.

        Construction is read-only with respect to history: existing records
        are rehydrated into the retrieval index and no commit is made.

        Args:
            data_dir: Root of the git-backed store (``records/`` and
                ``audit/`` live under it).
            materiality_threshold: Minimum number of replayed runs that must
                back a promotion (see module docstring).
            actor: Name recorded in the audit log for mutations issued by
                this engine instance.
        """
        if materiality_threshold < 1:
            raise ValueError("materiality_threshold must be >= 1")
        self.data_dir = Path(data_dir)
        self.materiality_threshold = materiality_threshold
        self.actor = actor
        self.records_dir = self.data_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(self.data_dir / "audit" / "audit.jsonl")
        self._init_git()
        self._records: dict[str, LearningRecord] = {}
        self._rehydrate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        """Return the top-``k`` promoted records relevant to ``task_context``.

        Only ``promoted`` records are ever returned — warm runs must see
        exactly the knowledge the memory gate has proven. Relevance is the
        v1 type/tag matching described in the module docstring (tracked
        open question); confidence orders within a relevance tier.
        """
        if k <= 0:
            return []
        context_tokens = _tokenize(task_context)
        context_text = " ".join(context_tokens)
        token_set = set(context_tokens)
        scored = [
            (self._relevance(record, token_set, context_text), record)
            for record in self._records.values()
            if record.status == _PROMOTED
        ]
        scored.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].id))
        return [record for _, record in scored[:k]]

    def propose(self, draft: LearningRecord | Mapping[str, Any]) -> LearningRecord:
        """Store ``draft`` as a new ``ephemeral`` record and return it.

        Proposing never fails on content or state: any mapping-shaped draft
        is coerced with generous defaults (missing body fields, provenance,
        confidence, id, and timestamps are filled in), a draft id that
        already exists lands under a fresh engine-generated id instead of
        overwriting, and no gating or quality filtering happens here —
        filtering is the memory gate's job. The one hard boundary is the
        pinned record schema: a draft that cannot be parsed as a
        :class:`LearningRecord` (illegal ``type``, out-of-range confidence)
        raises ``pydantic.ValidationError`` rather than corrupting the store.

        The draft's ``status`` is forced to ``ephemeral`` and its
        ``counterfactual`` is reset: replay evidence is filled by the gate,
        never claimed by the proposer.
        """
        record = self._coerce_draft(draft)
        self._write_record_file(record)
        self.audit.append(
            "propose",
            actor=self.actor,
            learning_id=record.id,
            evidence=f"status={record.status}; confidence={record.confidence}",
        )
        self._commit(f"propose: {record.id} (status={record.status})")
        self._records[record.id] = record
        return record

    def queue_replay(self, learning_id: str, run_ids: list[str]) -> None:
        """Attach a pending counterfactual replay (``run_ids``) to a record.

        Marks the record ``candidate``; the replay compares cold and warm
        verdicts on the recorded failure cases and its results feed the
        memory gate via :meth:`promote`.
        """
        record = self._get(learning_id)
        if record.status not in (_EPHEMERAL, _CANDIDATE):
            return self._refuse_transition(
                record,
                "queue-replay",
                f"cannot queue replay for record in status '{record.status}'",
            )
        merged = list(dict.fromkeys([*record.counterfactual.replay_run_ids, *run_ids]))
        updated = record.model_copy(
            update={
                "status": _CANDIDATE,
                "counterfactual": record.counterfactual.model_copy(
                    update={"replay_run_ids": merged}
                ),
            }
        )
        self._write_record_file(updated)
        self.audit.append(
            "queue-replay",
            actor=self.actor,
            learning_id=learning_id,
            evidence=f"run_ids={run_ids}",
        )
        self._commit(f"queue-replay: {learning_id} (status={_CANDIDATE}, runs={len(run_ids)})")
        self._records[learning_id] = updated

    def promote(
        self,
        learning_id: str,
        replay_evidence: RecordCounterfactual,
        *,
        reject_reason: str | None = None,
    ) -> LearningRecord:
        """Promote ``learning_id`` to ``promoted`` given passing replay evidence.

        Promotion happens only when ``replay_evidence`` shows a verdict
        change and its evidence set meets the engine's materiality
        threshold. ``reject_reason`` lets a gate refuse a promotion on a
        precondition of its own (e.g. the memory gate's promotion-time
        oracle-leak scan) while the engine still owns the write, the audit
        entry, and the commit. A rejected promotion leaves the record
        ``ephemeral``, records the rejection reason in the audit log,
        commits that audit entry, and returns the unchanged record — the
        audit log shows the gate worked. The stamped ``counterfactual`` on
        a promotion carries exactly the evidence the decision was made on.
        """
        record = self._get(learning_id)
        if record.status == _DROPPED:
            self._refuse_transition(
                record, "promote", "record is dropped (terminal); promotion refused"
            )
        if record.status == _PROMOTED:
            self._refuse_transition(record, "promote", "record is already promoted")
        runs = len(replay_evidence.replay_run_ids)
        evidence_summary = f"verdict_change={replay_evidence.verdict_change}; replayed_runs={runs}"
        if replay_evidence.verdict_change is not True:
            return self._reject_promotion(
                record,
                replay_evidence,
                reject_reason or f"replay evidence shows no verdict change ({evidence_summary})",
            )
        if runs < self.materiality_threshold:
            return self._reject_promotion(
                record,
                replay_evidence,
                f"evidence set of {runs} replayed run(s) is below the materiality threshold "
                f"({self.materiality_threshold})",
            )
        promoted = record.model_copy(
            update={
                "status": _PROMOTED,
                "promoted_at": _utcnow(),
                "counterfactual": replay_evidence.model_copy(),
            }
        )
        self._write_record_file(promoted)
        self.audit.append(
            "promote",
            actor=self.actor,
            learning_id=learning_id,
            evidence=f"{evidence_summary}; {replay_evidence.evidence or ''}".rstrip("; "),
        )
        self._commit(f"promote: {learning_id} (status={_PROMOTED})")
        self._records[learning_id] = promoted
        return promoted

    def demote(self, learning_id: str, reason: str) -> LearningRecord:
        """Move ``learning_id`` to ``dropped`` with ``reason`` recorded.

        Dropped records remain in the store for audit; the engine never
        silently forgets what it once believed. The reason is recorded in
        the audit log entry; only promoted records can be demoted.
        """
        if not reason or not reason.strip():
            raise ValueError("demote requires a non-empty reason")
        record = self._get(learning_id)
        if record.status != _PROMOTED:
            self._refuse_transition(
                record,
                "demote",
                f"cannot demote record in status '{record.status}' (only promoted records demote)",
            )
        dropped = record.model_copy(update={"status": _DROPPED})
        self._write_record_file(dropped)
        self.audit.append("demote", actor=self.actor, learning_id=learning_id, reason=reason)
        self._commit(f"demote: {learning_id} (status={_DROPPED}, reason: {reason})")
        self._records[learning_id] = dropped
        return dropped

    def get(self, learning_id: str) -> LearningRecord:
        """Return the stored record with ``learning_id`` (any status)."""
        return self._get(learning_id)

    # ------------------------------------------------------------------
    # Internal helpers (also used by engine.bootstrap)
    # ------------------------------------------------------------------

    def _ingest_batch(self, records: list[LearningRecord]) -> list[LearningRecord]:
        """Write ``records`` plus per-record audit lines, committed as one batch.

        Used by the bootstrap loader so a seed pack lands as a single
        bootstrap mutation; refuses ids already present in the store.
        """
        for record in records:
            if record.id in self._records:
                raise ValueError(f"record id '{record.id}' already exists in the store")
        for record in records:
            self._write_record_file(record)
            self.audit.append(
                "bootstrap-load",
                actor=self.actor,
                learning_id=record.id,
                evidence=f"seed; status={record.status}; confidence={record.confidence}",
            )
        self._commit(f"bootstrap: load {len(records)} promoted seed record(s)")
        for record in records:
            self._records[record.id] = record
        return list(records)

    # ------------------------------------------------------------------
    # Storage plumbing
    # ------------------------------------------------------------------

    def _rehydrate(self) -> None:
        """Load every stored record into the retrieval index.

        A corrupt record file is a store-integrity failure: raise rather
        than skip, so the damage surfaces instead of silently shrinking
        the engine's knowledge.
        """
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                record = LearningRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as error:
                message = f"corrupt record file: {path}: {error}"
                raise RuntimeError(message) from error
            self._records[record.id] = record

    def _get(self, learning_id: str) -> LearningRecord:
        try:
            return self._records[learning_id]
        except KeyError:
            raise RecordNotFoundError(f"no record with id '{learning_id}'") from None

    def _relevance(self, record: LearningRecord, token_set: set[str], context_text: str) -> int:
        tags = {record.type, record.type.replace("-", " ")}
        tags.update(token for token in _tokenize(record.title) if len(token) >= 3)
        return sum(1 for tag in tags if (tag in context_text if " " in tag else tag in token_set))

    def _coerce_draft(self, draft: LearningRecord | Mapping[str, Any]) -> LearningRecord:
        if isinstance(draft, LearningRecord):
            payload = draft.model_dump()
        elif isinstance(draft, Mapping):
            payload = dict(draft)
        else:
            raise TypeError("draft must be a LearningRecord or a mapping of record fields")
        draft_id = payload.get("id")
        if not draft_id or not _SAFE_ID.match(str(draft_id)) or str(draft_id) in self._records:
            payload["id"] = self._generate_id()
        payload["status"] = _EPHEMERAL
        payload["promoted_at"] = None
        payload["counterfactual"] = RecordCounterfactual().model_dump()
        if payload.get("confidence") is None:
            payload["confidence"] = 0.0
        if payload.get("created") is None:
            payload["created"] = _utcnow()
        if not payload.get("type"):
            # Bare drafts default to the reflector's most common output kind.
            payload["type"] = "failure-pattern"
        body = payload.get("body")
        if isinstance(body, Mapping):
            payload["body"] = {**{field: "" for field in RecordBody.model_fields}, **dict(body)}
        elif body is None:
            payload["body"] = {field: "" for field in RecordBody.model_fields}
        provenance = payload.get("provenance")
        if isinstance(provenance, Mapping):
            payload["provenance"] = {
                "proposed_by": "unknown",
                "origin": "agent-proposal",
                "source": "",
                "note": "provenance filled by engine: draft carried none",
                **dict(provenance),
            }
        elif provenance is None:
            payload["provenance"] = RecordProvenance(
                proposed_by="unknown",
                origin="agent-proposal",
                source="",
                note="provenance filled by engine: draft carried none",
            ).model_dump()
        return LearningRecord.model_validate(payload)

    def _generate_id(self) -> str:
        while True:
            candidate = f"lr-{_utcnow():%Y%m%d}-{uuid4().hex[:8]}"
            if candidate not in self._records:
                return candidate

    def _write_record_file(self, record: LearningRecord) -> None:
        path = self.records_dir / f"{record.id}.json"
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def _reject_promotion(
        self, record: LearningRecord, replay_evidence: RecordCounterfactual, reason: str
    ) -> LearningRecord:
        # Return the record to ephemeral per the v1 spec; strip the failed
        # evidence from the record — the audit log is its durable home.
        reverted = record.model_copy(
            update={"status": _EPHEMERAL, "counterfactual": RecordCounterfactual()}
        )
        self._write_record_file(reverted)
        self.audit.append(
            "promote-rejected",
            actor=self.actor,
            learning_id=record.id,
            reason=reason,
            evidence=(
                f"verdict_change={replay_evidence.verdict_change}; "
                f"replayed_runs={len(replay_evidence.replay_run_ids)}"
            ),
        )
        self._commit(f"reject-promotion: {record.id} (stays ephemeral: {reason})")
        self._records[record.id] = reverted
        return reverted

    def _refuse_transition(self, record: LearningRecord, action: str, reason: str) -> None:
        self.audit.append(f"{action}-rejected", actor=self.actor, learning_id=record.id, reason=reason)
        self._commit(f"reject-{action}: {record.id} ({reason})")
        raise InvalidTransitionError(reason)

    def _run_git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(self.data_dir), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,  # inspected below — a nonzero code raises RuntimeError
        )
        if proc.returncode != 0:
            message = f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
            raise RuntimeError(message)
        return proc.stdout

    def _init_git(self) -> None:
        if not (self.data_dir / ".git").exists():
            self._run_git("init", "--quiet")
        # Commits need an identity; set one locally when the environment
        # has none (CI runners often do not).
        for key, value in (("user.name", "evogine-engine"), ("user.email", "engine@evogine.local")):
            probe = subprocess.run(
                ["git", "-C", str(self.data_dir), "config", key],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,  # a missing key is expected; probed via returncode/output
            )
            if probe.returncode != 0 or not probe.stdout.strip():
                self._run_git("config", key, value)

    def _commit(self, message: str) -> None:
        self._run_git("add", "-A", "records", "audit")
        self._run_git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message)
