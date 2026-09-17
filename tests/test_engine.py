"""Engine core tests: lifecycle, audit trail, git-backed storage, retrieval."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from evogine.engine.audit import AuditLog
from evogine.engine.records import RecordCounterfactual
from evogine.engine.store import Engine, InvalidTransitionError, RecordNotFoundError


def make_engine(tmp_path: Path, **kwargs) -> Engine:
    return Engine(tmp_path / "engine", **kwargs)


def draft(**overrides) -> dict:
    payload = {
        "id": "learn_test_0001",
        "type": "failure-pattern",
        "title": "Retry storms from unbounded 429 backoff",
        "body": {
            "signal": "rising 429 counts",
            "failure_mode": "retries amplify rate limits",
            "detection": "counterfactual replay flips the verdict",
            "fix_strategy": "jittered exponential backoff",
        },
        "provenance": {
            "proposed_by": "reflector",
            "origin": "rollout",
            "source": "rollout-2791",
            "note": "test draft",
        },
    }
    payload.update(overrides)
    return payload


def passing_evidence(runs: int = 2) -> RecordCounterfactual:
    return RecordCounterfactual(
        replay_run_ids=[f"run_{i}" for i in range(runs)],
        verdict_change=True,
        evidence="warm verdicts flip FAIL -> PASS on every replayed run",
    )


def git_subjects(engine: Engine) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(engine.data_dir), "log", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.splitlines()


def commit_count(engine: Engine) -> int:
    proc = subprocess.run(
        ["git", "-C", str(engine.data_dir), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def test_propose_lands_ephemeral_with_provenance(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    record = engine.propose(draft())
    assert record.status == "ephemeral"
    assert record.promoted_at is None
    assert record.provenance.proposed_by == "reflector"
    assert (Path(engine.data_dir) / "records" / f"{record.id}.json").exists()
    assert engine.audit.find("propose", record.id)
    assert commit_count(engine) == 1


def test_propose_fills_missing_fields_with_defaults(tmp_path: Path) -> None:
    """A bare draft still lands: defaults are filled, provenance stays present."""
    engine = make_engine(tmp_path)
    record = engine.propose({"title": "Bare draft"})
    assert record.status == "ephemeral"
    assert record.id
    assert record.provenance.proposed_by == "unknown"
    assert record.confidence == 0.0
    assert record.body.signal == ""
    assert record.counterfactual.verdict_change is None
    assert engine.get(record.id) == record


def test_propose_never_promotes_or_carries_self_claimed_evidence(tmp_path: Path) -> None:
    """A proposer cannot self-promote: status is forced ephemeral, evidence reset."""
    engine = make_engine(tmp_path)
    record = engine.propose(
        draft(
            status="promoted",
            counterfactual={
                "replay_run_ids": ["run_1"],
                "verdict_change": True,
                "evidence": "trust me",
            },
        )
    )
    assert record.status == "ephemeral"
    assert record.counterfactual.verdict_change is None
    assert record.counterfactual.replay_run_ids == []


def test_propose_duplicate_id_never_overwrites(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    first = engine.propose(draft())
    second = engine.propose(draft(title="same id, new draft"))
    assert second.id != first.id
    assert engine.get(first.id).title == "Retry storms from unbounded 429 backoff"
    assert engine.get(second.id).title == "same id, new draft"


def test_propose_rejects_unsafe_id_instead_of_escaping_the_store(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    record = engine.propose(draft(id="../escape"))
    assert "/" not in record.id
    assert record.id != "../escape"


def test_propose_promote_flow_with_replay_evidence(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    proposed = engine.propose(draft())
    promoted = engine.promote(proposed.id, passing_evidence())
    assert promoted.status == "promoted"
    assert promoted.promoted_at is not None
    assert promoted.counterfactual.verdict_change is True
    assert promoted.counterfactual.replay_run_ids == ["run_0", "run_1"]
    retrieved = engine.retrieve("failure pattern retry storms", k=5)
    assert [r.id for r in retrieved] == [proposed.id]
    assert engine.audit.find("promote", proposed.id)


def test_promote_rejected_when_verdict_change_false(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    proposed = engine.propose(draft())
    evidence = RecordCounterfactual(
        replay_run_ids=["run_1"], verdict_change=False, evidence="no verdict moved"
    )
    result = engine.promote(proposed.id, evidence)
    assert result.status == "ephemeral"
    assert result.promoted_at is None
    assert engine.get(proposed.id).status == "ephemeral"
    rejections = engine.audit.find("promote-rejected", proposed.id)
    assert len(rejections) == 1
    assert "no verdict change" in rejections[0].reason
    assert any("reject-promotion" in subject for subject in git_subjects(engine))


def test_promote_rejected_below_materiality_threshold(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, materiality_threshold=3)
    proposed = engine.propose(draft())
    result = engine.promote(proposed.id, passing_evidence(runs=1))
    assert result.status == "ephemeral"
    rejection = engine.audit.find("promote-rejected", proposed.id)[0]
    assert "materiality threshold" in rejection.reason


def test_promote_rejection_returns_queued_record_to_ephemeral(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    proposed = engine.propose(draft())
    engine.queue_replay(proposed.id, ["run_1"])
    assert engine.get(proposed.id).status == "candidate"
    result = engine.promote(
        proposed.id,
        RecordCounterfactual(replay_run_ids=["run_1"], verdict_change=False, evidence="no flip"),
    )
    assert result.status == "ephemeral"
    assert engine.get(proposed.id).counterfactual.replay_run_ids == []


def test_queue_replay_marks_candidate_and_accumulates_runs(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    proposed = engine.propose(draft())
    engine.queue_replay(proposed.id, ["run_1", "run_2"])
    engine.queue_replay(proposed.id, ["run_2", "run_3"])
    record = engine.get(proposed.id)
    assert record.status == "candidate"
    assert record.counterfactual.replay_run_ids == ["run_1", "run_2", "run_3"]
    assert commit_count(engine) == 3


def test_demote_flow_drops_promoted_record_with_reason(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    proposed = engine.propose(draft())
    engine.promote(proposed.id, passing_evidence())
    dropped = engine.demote(proposed.id, "regressed run_42 after adoption")
    assert dropped.status == "dropped"
    assert engine.get(proposed.id).status == "dropped"
    demotions = engine.audit.find("demote", proposed.id)
    assert len(demotions) == 1
    assert demotions[0].reason == "regressed run_42 after adoption"
    assert engine.retrieve("failure pattern retry storms", k=5) == []


def test_demote_requires_promoted_record(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    proposed = engine.propose(draft())
    with pytest.raises(InvalidTransitionError, match="ephemeral"):
        engine.demote(proposed.id, "not promoted yet")
    refusal = engine.audit.find("demote-rejected", proposed.id)
    assert len(refusal) == 1


def test_demote_requires_a_reason(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    with pytest.raises(ValueError, match="reason"):
        engine.demote("learn_test_0001", "  ")


def test_unknown_learning_id_raises_record_not_found(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    with pytest.raises(RecordNotFoundError):
        engine.get("learn_missing")
    with pytest.raises(RecordNotFoundError):
        engine.promote("learn_missing", passing_evidence())
    with pytest.raises(RecordNotFoundError):
        engine.queue_replay("learn_missing", ["run_1"])


def test_retrieve_never_returns_ephemeral_or_dropped_records(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    kept = engine.propose(draft(id="learn_kept"))
    ephemeral = engine.propose(draft(id="learn_ephemeral", title="still unproven"))
    engine.promote(kept.id, passing_evidence())
    other = engine.propose(draft(id="learn_dropped", title="proven once, regressed later"))
    engine.promote(other.id, passing_evidence())
    engine.demote(other.id, "regressed a later run")
    results = engine.retrieve("failure pattern retry storms unproven", k=10)
    ids = [r.id for r in results]
    assert ephemeral.id not in ids
    assert other.id not in ids
    assert kept.id in ids
    assert all(r.status == "promoted" for r in results)


def test_retrieve_ranks_by_relevance_then_confidence(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    low_conf_match = engine.propose(
        draft(id="learn_playbook_low", type="playbook", title="playbook for retry storms", confidence=0.3)
    )
    high_conf_match = engine.propose(
        draft(id="learn_playbook_high", type="playbook", title="playbook for retry storms", confidence=0.9)
    )
    non_match = engine.propose(
        draft(id="learn_check", type="check", title="freeze the clock at DST boundaries")
    )
    engine.promote(low_conf_match.id, passing_evidence())
    engine.promote(high_conf_match.id, passing_evidence())
    engine.promote(non_match.id, passing_evidence())
    ranked = engine.retrieve("need a playbook for retry storms", k=3)
    assert [r.id for r in ranked][:2] == ["learn_playbook_high", "learn_playbook_low"]
    assert ranked[0].confidence >= ranked[1].confidence


def test_retrieve_falls_back_to_confidence_when_nothing_matches(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    first = engine.propose(draft(id="learn_first", confidence=0.9))
    second = engine.propose(draft(id="learn_second", confidence=0.4))
    engine.promote(first.id, passing_evidence())
    engine.promote(second.id, passing_evidence())
    results = engine.retrieve("totally unrelated task context", k=1)
    assert [r.id for r in results] == ["learn_first"]


def test_retrieve_respects_k_and_empty_store(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    assert engine.retrieve("anything", k=5) == []
    for i in range(3):
        record = engine.propose(draft(id=f"learn_k_{i}"))
        engine.promote(record.id, passing_evidence())
    assert len(engine.retrieve("failure pattern", k=2)) == 2
    assert engine.retrieve("failure pattern", k=0) == []


def test_audit_log_records_required_fields(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    record = engine.propose(draft())
    entries = AuditLog(Path(engine.data_dir) / "audit" / "audit.jsonl").entries()
    proposed = [e for e in entries if e.action == "propose" and e.learning_id == record.id]
    assert len(proposed) == 1
    entry = proposed[0]
    assert entry.timestamp is not None
    assert entry.actor == "engine"
    assert entry.learning_id == record.id


def test_git_log_shows_one_commit_per_mutation(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    record = engine.propose(draft())
    engine.queue_replay(record.id, ["run_1", "run_2"])
    engine.promote(record.id, passing_evidence())
    engine.demote(record.id, "regressed a later run")
    assert commit_count(engine) == 4
    subjects = git_subjects(engine)
    assert subjects[0] == f"demote: {record.id} (status=dropped, reason: regressed a later run)"
    assert subjects[1] == f"promote: {record.id} (status=promoted)"
    assert subjects[2] == f"queue-replay: {record.id} (status=candidate, runs=2)"
    assert subjects[3] == f"propose: {record.id} (status=ephemeral)"
    # A rejected promotion is itself exactly one committed mutation.
    other = engine.propose(draft(id="learn_rejected"))
    engine.promote(
        other.id,
        RecordCounterfactual(replay_run_ids=["run_1"], verdict_change=False, evidence="no flip"),
    )
    assert commit_count(engine) == 6
    assert git_subjects(engine)[0].startswith(f"reject-promotion: {other.id}")


def test_engine_rehydrates_records_from_disk(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    record = engine.propose(draft())
    engine.promote(record.id, passing_evidence())
    reopened = Engine(engine.data_dir)
    assert reopened.get(record.id).status == "promoted"
    assert [r.id for r in reopened.retrieve("failure pattern retry storms", k=5)] == [record.id]


def test_engine_refuses_corrupt_record_files(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    corrupt = Path(engine.data_dir) / "records" / "learn_corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="learn_corrupt"):
        Engine(engine.data_dir)


def test_materiality_threshold_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="materiality_threshold"):
        make_engine(tmp_path, materiality_threshold=0)


def test_draft_schema_violations_surface(tmp_path: Path) -> None:
    """The pinned schema is the one boundary proposing does not absorb."""
    engine = make_engine(tmp_path)
    with pytest.raises(ValidationError):
        engine.propose(draft(type="gossip"))
    with pytest.raises(ValidationError):
        engine.propose(draft(confidence=1.5))
