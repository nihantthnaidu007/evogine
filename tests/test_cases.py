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

import pytest

from evogine.bench.gate import PropertiesGate
from evogine.bench.task import discover_case_ids, load_task

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
