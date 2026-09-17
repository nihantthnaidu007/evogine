"""The standing oracle-leak guarantee (ported from Darwin's test_oracle_leak).

The candidate builder must never see the oracle: ``properties.json`` is the
hidden oracle, checked host-side by the gate, and no expected value from it
may appear in any generator-facing surface — the template constants in
``bench/prompts.py``, the rendered builder messages for any case, any case's
``scenario.md`` (surfaced as ``scenario_brief``), or its meta description.

Like Darwin's version, the guard tests REAL surfaces and proves it is not
vacuous three ways: every case must contribute scannable oracle strings, the
scanned surfaces are the ones the harness actually renders, and a planted
leak of a REAL oracle value into a REAL surface must be caught by the same
matching logic the real assertions use.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from evogine.bench import prompts
from evogine.bench.prompts import build_builder_messages
from evogine.bench.task import BenchTask, discover_case_ids, load_task

CASE_IDS = discover_case_ids()

# Scalar oracle strings shorter than this ("true", "0", "[]") match prose
# coincidentally and carry no exploitable signal; skipping them keeps the
# scan free of false positives without weakening it. Long scalars — dates,
# ids, exception names — and every composite value are scanned.
MIN_ORACLE_LENGTH = 5


def _oracle_strings(value: Any) -> list[str]:
    """Scannable oracle strings for one expected value."""
    strings = [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    if isinstance(value, str):
        strings.append(value)
    return [s for s in strings if len(s) >= MIN_ORACLE_LENGTH]


def _case_oracle_strings(case_id: str, kind: str = "expected") -> list[str]:
    """Frozen oracle strings of one kind from one case's properties.json.

    ``kind="expected"`` collects golden output values; ``kind="raises"``
    collects raised exception type names.
    """
    task = load_task(case_id)
    strings: list[str] = []
    for prop in task.properties["properties"]:
        for row in prop["rows"]:
            for holder in (row, *row.get("steps", [])):
                if kind == "raises" and "raises" in holder:
                    strings.extend(_oracle_strings(holder["raises"]))
                if kind == "expected" and "expected" in holder:
                    strings.extend(_oracle_strings(holder["expected"]))
    return strings


def _all_oracle_strings() -> list[str]:
    return [
        oracle
        for case_id in CASE_IDS
        for oracle in (
            _case_oracle_strings(case_id, "expected")
            + _case_oracle_strings(case_id, "raises")
        )
    ]


def _descriptive_surfaces(task: BenchTask) -> list[str]:
    """Task-descriptive material: what the case ASKS for, not the source."""
    return [task.scenario_brief, task.description]


def _builder_surfaces(task: BenchTask) -> list[str]:
    """Every text the builder can see for one case, harness-rendered."""
    messages = build_builder_messages(task, [])
    return [task.scenario_brief, task.description, *(m["content"] for m in messages)]


def _prompt_template_strings() -> list[str]:
    return [
        value
        for name, value in vars(prompts).items()
        if name.startswith("BUILDER_") and isinstance(value, str)
    ]


def _find_leaks(surfaces: list[str], oracle_strings: list[str]) -> list[str]:
    """Oracle strings that appear verbatim in any surface."""
    return [
        oracle
        for oracle in oracle_strings
        for surface in surfaces
        if oracle in surface
    ]


def test_every_case_has_a_real_oracle() -> None:
    # Non-vacuous precondition at the corpus level: some case must have
    # scannable oracle strings, and every case must freeze a properties
    # oracle at all (even one whose values are all short scalars — its
    # surface scan then runs over an empty scannable set, by construction).
    assert _all_oracle_strings(), "no scannable oracle strings in the corpus"
    for case_id in CASE_IDS:
        task = load_task(case_id)
        assert task.properties["properties"], f"{case_id}: no properties"


def test_builder_templates_are_oracle_free() -> None:
    oracle_strings = _all_oracle_strings()
    assert oracle_strings
    templates = _prompt_template_strings()
    assert templates, "no prompt template constants found in bench.prompts"
    assert not _find_leaks(templates, oracle_strings)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_case_surfaces_are_oracle_free(case_id: str) -> None:
    """Golden output values never appear in ANY builder-facing surface.

    Raise-type names are scanned against descriptive material only
    (scenario.md, meta description, template constants): the candidate's own
    current source legitimately names the exception types it already raises
    for contract preservation, so a bare "ValueError" there is not oracle
    signal — but "OverflowError" named in a scenario would hand over the fix.
    """
    task = load_task(case_id)
    expected_oracles = [
        oracle
        for other in CASE_IDS
        for oracle in _case_oracle_strings(other, "expected")
    ]
    raise_oracles = [
        oracle for other in CASE_IDS for oracle in _case_oracle_strings(other, "raises")
    ]
    leaks = _find_leaks(_builder_surfaces(task), expected_oracles)
    leaks += _find_leaks(
        _descriptive_surfaces(task) + _prompt_template_strings(), raise_oracles
    )
    assert not leaks, f"{case_id}: builder-facing surfaces leak oracle values: {leaks}"


def test_scan_catches_a_planted_leak() -> None:
    # The tester must be able to fail: plant a REAL oracle value from the
    # first case into a REAL surface of another case and assert the same
    # matching logic the real assertions use reports it.
    planted_case, surface_case = CASE_IDS[0], CASE_IDS[-1]
    planted = _case_oracle_strings(planted_case, "expected")[0]
    task = load_task(surface_case)
    surfaces = _builder_surfaces(task)
    surfaces.append(f"pressure note: the expected outcome is {planted}")
    assert _find_leaks(surfaces, _all_oracle_strings()), (
        "the leak scan failed to catch a planted oracle value — it is blind"
    )
