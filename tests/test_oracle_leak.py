"""The standing oracle-leak guarantee (ported from Darwin's test_oracle_leak).

The builder must never see the oracle: ``properties.json`` is the hidden
oracle, checked host-side by the gate, and no expected value from it may
appear in any generator-facing surface — the prompt constants in
``swarm/roles.py``, the messages the three roles actually render for any
case, the retrieval-context assembly (``render_knowledge``) that carries
promoted learnings into the builder prompt, any case's ``scenario.md``
(surfaced as ``scenario_brief``), or its meta description. Gate test names
(``def test_*`` in each case's host-side ``test_suite.py``) are scanned too:
a prompt or a promoted learning quoting them hands over the gate's hidden
behavior.

The scan watches the LIVE prompt path only. ``bench/prompts.py`` stopped
being one when PR #8 wired the loop to the swarm roles: nothing under
``src/`` imports it (this test was its only importer), so scanning it
verified a file the builder/verifier/reflector never consume — false
assurance, not a guard.

Like Darwin's version, the guard tests REAL surfaces and proves it is not
vacuous three ways: every case must contribute scannable oracle strings, the
scanned surfaces are the ones the roles actually render, and a planted leak
of a REAL oracle value — and of a REAL test name riding a poisoned learning
— into the live path must be caught by the same matching logic the real
assertions use.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evogine.bench.task import BenchTask, discover_case_ids, load_task
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.swarm import roles

CASE_IDS = discover_case_ids()

#: The bootstrap seed pack, read directly (no store): it is what ``Engine``
#: rehydrates into retrieval, so it is what the warm arm's retrieval-context
#: assembly can render into a builder prompt.
SEEDS_DIR = Path(roles.__file__).resolve().parent.parent / "seeds"

# Scalar oracle strings shorter than this ("true", "0", "[]") match prose
# coincidentally and carry no exploitable signal; skipping them keeps the
# scan free of false positives without weakening it. Long scalars — dates,
# ids, exception names — and every composite value are scanned.
MIN_ORACLE_LENGTH = 5

#: Prompt-constant naming convention in ``swarm/roles.py``: role-prefixed
#: upper-case string constants. New role prompts following the convention
#: are scanned automatically.
_ROLE_TEMPLATE_PREFIXES = ("BUILDER_", "VERIFIER_", "REFLECTOR_")

#: Reflector evidence stands in for gate failure details, which are
#: oracle-free by construction (the code gate's rejection payloads never
#: carry expected values). A fixed benign string keeps the standing scan
#: deterministic.
BENIGN_EVIDENCE = "recorded outcome: the candidate lost to the baseline (verdict fail)."

_TEST_NAME_RE = re.compile(r"def (test_[A-Za-z0-9_]+)\(")


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


def _test_names() -> list[str]:
    """Gate test names from every case's host-side ``test_suite.py``."""
    names: list[str] = []
    for case_id in CASE_IDS:
        task = load_task(case_id)
        names.extend(_TEST_NAME_RE.findall(task.suite_path.read_text(encoding="utf-8")))
    return names


def _descriptive_surfaces(task: BenchTask) -> list[str]:
    """Task-descriptive material: what the case ASKS for, not the source."""
    return [task.scenario_brief, task.description]


def _role_template_strings() -> list[str]:
    """Every prompt constant in the live roles module, discovered by name."""
    return [
        value
        for name, value in sorted(vars(roles).items())
        if name.startswith(_ROLE_TEMPLATE_PREFIXES) and isinstance(value, str)
    ]


def _source_blob(task: BenchTask) -> str:
    """The baseline source rendered the way the live roles render it."""
    return "\n".join(
        f"### {relative}\n{source}" for relative, source in sorted(task.baseline_files.items())
    )


def _live_role_surfaces(task: BenchTask) -> list[str]:
    """Every text the live roles actually render for one case.

    Mirrors the loop's composition: builder messages (cold arm — no
    knowledge; the warm arm's knowledge block is scanned on its own), the
    verifier over a deterministic candidate stand-in (the baseline blob —
    real candidates are generated from these same oracle-free inputs), and
    the reflector over benign evidence.
    """
    builder = roles.build_builder_messages(task, [])
    verifier = roles.build_verifier_messages(task, _source_blob(task))
    reflector = roles.build_reflector_messages(task, BENIGN_EVIDENCE)
    return [
        task.scenario_brief,
        task.description,
        *(message["content"] for message in (*builder, *verifier, *reflector)),
    ]


def _seed_records() -> list[LearningRecord]:
    """The seed pack validated against the pinned learning-record schema."""
    files = sorted(SEEDS_DIR.glob("*.json"))
    assert files, "seed pack is empty — the warm arm has no knowledge to scan"
    return [
        LearningRecord.model_validate_json(seed_file.read_text(encoding="utf-8"))
        for seed_file in files
    ]


def _knowledge_surface() -> str:
    """The retrieval-context block exactly as the live builder renders it."""
    return roles.render_knowledge(_seed_records())


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
    # scannable oracle strings, every case must freeze a properties
    # oracle at all (even one whose values are all short scalars — its
    # surface scan then runs over an empty scannable set, by construction),
    # and the suites must expose gate test names for the name scan.
    assert _all_oracle_strings(), "no scannable oracle strings in the corpus"
    assert _test_names(), "no gate test names discovered in the case suites"
    for case_id in CASE_IDS:
        task = load_task(case_id)
        assert task.properties["properties"], f"{case_id}: no properties"


def test_role_templates_are_oracle_free() -> None:
    oracle_strings = _all_oracle_strings()
    assert oracle_strings
    templates = _role_template_strings()
    assert templates, "no prompt template constants found in evogine.swarm.roles"
    leaks = _find_leaks(templates, oracle_strings + _test_names())
    assert not leaks, f"role prompt templates leak oracle material: {leaks}"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_live_role_messages_are_oracle_free(case_id: str) -> None:
    """Golden output values and gate test names never appear in ANY live
    role surface; raise-type names never appear in descriptive material.

    Raise-type names are scanned against descriptive material only (the
    templates have their own test): the candidate's own source legitimately
    names the exception types it already raises for contract preservation,
    so a bare "ValueError" there is not oracle signal — but "OverflowError"
    named in a scenario would hand over the fix.
    """
    task = load_task(case_id)
    expected_oracles = [
        oracle for other in CASE_IDS for oracle in _case_oracle_strings(other, "expected")
    ]
    raise_oracles = [
        oracle for other in CASE_IDS for oracle in _case_oracle_strings(other, "raises")
    ]
    surfaces = _live_role_surfaces(task)
    leaks = _find_leaks(surfaces, expected_oracles + _test_names())
    leaks += _find_leaks(_descriptive_surfaces(task), raise_oracles)
    assert not leaks, f"{case_id}: live role surfaces leak oracle material: {leaks}"


def test_seeded_retrieval_context_is_oracle_free() -> None:
    """The seed pack, rendered through the live knowledge assembly, is clean.

    ``render_knowledge`` is the one channel from the learning store into a
    builder prompt: whatever a promoted record quotes travels verbatim into
    it, so the whole pack is scanned in the exact rendered form it reaches
    the builder in.
    """
    knowledge = _knowledge_surface()
    assert knowledge, "render_knowledge produced an empty block for the seed pack"
    leaks = _find_leaks([knowledge], _all_oracle_strings() + _test_names())
    assert not leaks, f"seed pack leaks oracle material into retrieval context: {leaks}"


def test_scan_catches_a_planted_leak() -> None:
    # The tester must be able to fail: plant a REAL oracle value from the
    # first case into a REAL live surface of another case and assert the
    # same matching logic the real assertions use reports it.
    planted_case, surface_case = CASE_IDS[0], CASE_IDS[-1]
    planted = _case_oracle_strings(planted_case, "expected")[0]
    task = load_task(surface_case)
    surfaces = _live_role_surfaces(task)
    surfaces.append(f"pressure note: the expected outcome is {planted}")
    assert _find_leaks(surfaces, _all_oracle_strings()), (
        "the leak scan failed to catch a planted oracle value — it is blind"
    )


def test_scan_catches_a_seeded_learning_quoting_oracle_material() -> None:
    """A promoted learning quoting an oracle value or gate test name is caught.

    The reflector drafts learnings and the memory gate promotes them, but a
    poisoned record reaches the builder only through the live
    retrieval-context assembly — so the proof plants one there and renders
    the actual builder prompt the loop would send.
    """
    planted_oracle = _case_oracle_strings(CASE_IDS[0], "expected")[0]
    planted_test_name = _test_names()[0]
    poisoned = LearningRecord(
        id="lr-guard-probe",
        type="failure-pattern",
        title=f"Always satisfy {planted_test_name}",
        body=RecordBody(
            signal="leak probe",
            failure_mode="leak probe",
            detection="leak probe",
            fix_strategy=f"hardcode the expected value {planted_oracle}",
        ),
        status="promoted",
        confidence=1.0,
        provenance=RecordProvenance(
            proposed_by="guard-test",
            origin="test",
            source="planted by test_scan_catches_a_seeded_learning_quoting_oracle_material",
            note="synthetic poisoned record; never stored",
        ),
        counterfactual=RecordCounterfactual(),
        created=datetime(2026, 9, 17, tzinfo=UTC),
    )
    task = load_task(CASE_IDS[0])
    messages = roles.build_builder_messages(task, [poisoned])
    rendered = [message["content"] for message in messages]
    rendered.append(roles.render_knowledge([poisoned]))
    oracle_hits = _find_leaks(rendered, _all_oracle_strings())
    test_name_hits = _find_leaks(rendered, _test_names())
    assert oracle_hits and test_name_hits, (
        "the leak scan failed to catch a seeded learning quoting oracle "
        "material — the retrieval-context assembly is unguarded"
    )
