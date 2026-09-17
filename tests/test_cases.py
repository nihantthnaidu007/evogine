"""Corpus-wide invariants for the decay-pressure case set.

Three per-case invariants define a valid decay-pressure case:

1. the reference passes every visible test and every hidden property — it is
   the clean target under pressure;
2. the degraded baseline passes every visible test — it looks fine on the
   happy path, exactly like the v1 it models;
3. the degraded baseline fails at least one hidden property — pressure
   exposes what the visible suite misses.

Corpus-shape tests pin the design brief: at least six distinct decay axes
and at least two multi-file (port-style) package cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evogine.bench.gate import PropertiesGate
from evogine.bench.task import UnsafeCaseId, discover_case_ids, load_task

CASE_IDS = discover_case_ids()
MIN_DISTINCT_AXES = 6
MIN_PACKAGE_CASES = 2


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_reference_passes_all_properties(case_id: str) -> None:
    task = load_task(case_id)
    outcome = PropertiesGate().evaluate(task.reference_files, task)
    assert outcome.verdict == "pass", (
        f"{case_id}: the reference must pass its own properties\n"
        f"{outcome.failure_detail}"
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_degraded_passes_visible_suite(case_id: str) -> None:
    task = load_task(case_id)
    outcome = PropertiesGate().evaluate(task.baseline_files, task)
    assert outcome.visible_failures == 0, (
        f"{case_id}: the degraded baseline must pass every visible test\n"
        f"{outcome.failure_detail}"
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_degraded_fails_hidden_properties(case_id: str) -> None:
    task = load_task(case_id)
    outcome = PropertiesGate().evaluate(task.baseline_files, task)
    assert outcome.properties_passed < outcome.properties_total, (
        f"{case_id}: the degraded baseline must fail at least one hidden "
        "property — without that, the case exerts no decay pressure"
    )


def test_corpus_has_cases() -> None:
    assert len(CASE_IDS) >= 10, f"corpus has only {len(CASE_IDS)} cases"


def test_corpus_spans_enough_axes() -> None:
    axes = {load_task(case_id).decay_axis for case_id in CASE_IDS}
    assert len(axes) >= MIN_DISTINCT_AXES, (
        f"corpus spans only {len(axes)} distinct decay axes: {sorted(axes)}"
    )


def test_corpus_has_package_cases() -> None:
    layouts = [load_task(case_id).layout for case_id in CASE_IDS]
    assert layouts.count("package") >= MIN_PACKAGE_CASES, (
        "corpus needs at least two multi-file (port-style) package cases"
    )


def test_every_case_has_visible_tests() -> None:
    for case_id in CASE_IDS:
        task = load_task(case_id)
        suite = task.suite_path.read_text(encoding="utf-8")
        assert "def test_" in suite, f"{case_id}: visible suite has no tests"


# Case-id containment: the join behind load_task must refuse escapes (the
# resolve-under-root pattern from gates/sandbox.py).


def test_load_task_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(UnsafeCaseId, match="bare directory name"):
        load_task("../seeds", cases_root=tmp_path)
    with pytest.raises(UnsafeCaseId, match="bare directory name"):
        load_task("a/b", cases_root=tmp_path)


def test_load_task_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(UnsafeCaseId, match="bare directory name"):
        load_task("/etc/passwd", cases_root=tmp_path)


def test_load_task_rejects_empty_or_null_case_id(tmp_path: Path) -> None:
    with pytest.raises(UnsafeCaseId, match="empty or contains a null byte"):
        load_task("", cases_root=tmp_path)
    with pytest.raises(UnsafeCaseId, match="empty or contains a null byte"):
        load_task("ca\x00se", cases_root=tmp_path)


def test_load_task_rejects_dot_and_dotdot(tmp_path: Path) -> None:
    with pytest.raises(UnsafeCaseId, match="bare directory name"):
        load_task(".", cases_root=tmp_path)
    with pytest.raises(UnsafeCaseId, match="bare directory name"):
        load_task("..", cases_root=tmp_path)


def test_load_task_rejects_symlink_escape(tmp_path: Path) -> None:
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    cases_root = tmp_path / "cases"
    cases_root.mkdir()
    (cases_root / "innocent").symlink_to(decoy)
    with pytest.raises(UnsafeCaseId, match="escapes the cases root"):
        load_task("innocent", cases_root=cases_root)


def test_load_task_nonexistent_case_still_fails_loudly(tmp_path: Path) -> None:
    # A well-formed unknown id passes containment and fails later, with the
    # ordinary missing-file error — validation narrows the join, not the
    # loud-failure contract.
    with pytest.raises(ValueError, match="missing case file"):
        load_task("no_such_case", cases_root=tmp_path)


def test_load_task_still_loads_valid_case() -> None:
    task = load_task(CASE_IDS[0])
    assert task.case_id == CASE_IDS[0]
