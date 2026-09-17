"""Benchmark task model: loads a decay-pressure case from ``bench/cases/<id>/``.

A case directory is the unit of benchmark work:

.. code-block:: text

    bench/cases/<case_id>/
        reference.py      (or reference/ package)  the clean, correct target — host-side only
        degraded.py       (or degraded/ package)   the plausible v1 that decays — the builder's start
        test_suite.py     visible happy-path tests the degraded version PASSES
        properties.json   hidden invariants, checked host-side by the gate, never prompted
        meta.json         case_id, decay_axis, difficulty, description
        scenario.md       what pressure breaks it — generator-visible, oracle-free

The decay-pressure premise: a baseline that looks fine on the happy path and
fails under pressure. The visible suite pins the happy path (so the degraded
baseline passes it); the hidden properties pin the contract under pressure
(so the degraded baseline fails at least one of them). Loading is validated
loudly: a malformed case raises instead of silently shrinking the corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent / "cases"
REQUIRED_META_FIELDS = ("case_id", "decay_axis", "difficulty", "description")
VALID_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class BenchTask:
    """One decay-pressure task, loaded from its case directory.

    ``description`` and ``scenario_brief`` are generator-visible case
    material. ``properties`` is the hidden oracle: host-side only, never
    rendered into a prompt (enforced by tests/test_oracle_leak.py).
    ``baseline_files`` is the degraded starting point expressed as candidate
    files; ``reference_files`` is the clean target, host-side only.
    """

    case_id: str
    root: Path
    layout: str  # "single" or "package"
    decay_axis: str
    difficulty: str
    description: str
    scenario_brief: str
    properties: dict
    suite_path: Path
    baseline_files: dict[str, str]
    reference_files: dict[str, str]


def discover_case_ids(cases_root: Path | None = None) -> list[str]:
    """Return the sorted ids of every well-formed-enough case directory.

    A directory counts when it carries a meta.json; deeper validation happens
    in :func:`load_task` so a broken case is named, not silently skipped.
    """
    root = cases_root if cases_root is not None else CASES_DIR
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if (path / "meta.json").is_file())


def load_task(case_id: str, cases_root: Path | None = None) -> BenchTask:
    """Load and validate the case ``case_id``; raise on any malformation."""
    root = (cases_root if cases_root is not None else CASES_DIR) / case_id
    meta = _load_json(root / "meta.json")
    missing = [name for name in REQUIRED_META_FIELDS if name not in meta]
    if missing:
        raise ValueError(f"{case_id}: meta.json missing fields {missing}")
    if meta["difficulty"] not in VALID_DIFFICULTIES:
        raise ValueError(f"{case_id}: bad difficulty {meta['difficulty']!r}")

    properties = _load_json(root / "properties.json")
    if properties.get("case_id") != case_id:
        raise ValueError(f"{case_id}: properties.json case_id mismatch")
    if "entry" not in properties or not isinstance(properties.get("properties"), list):
        raise ValueError(f"{case_id}: properties.json needs 'entry' and a 'properties' list")

    layout = _infer_layout(case_id, root)
    return BenchTask(
        case_id=case_id,
        root=root,
        layout=layout,
        decay_axis=meta["decay_axis"],
        difficulty=meta["difficulty"],
        description=meta["description"],
        scenario_brief=(root / "scenario.md").read_text(encoding="utf-8"),
        properties=properties,
        suite_path=root / "test_suite.py",
        baseline_files=_source_files(root, "degraded", layout),
        reference_files=_source_files(root, "reference", layout),
    )


def _infer_layout(case_id: str, root: Path) -> str:
    if (root / "degraded.py").is_file():
        layout = "single"
    elif (root / "degraded").is_dir():
        layout = "package"
    else:
        raise ValueError(f"{case_id}: no degraded.py or degraded/ package found")
    reference_ok = (root / "reference.py").is_file() or (root / "reference").is_dir()
    if not reference_ok:
        raise ValueError(f"{case_id}: no reference.py or reference/ package found")
    return layout


def _source_files(root: Path, side: str, layout: str) -> dict[str, str]:
    """Collect one side of a case as candidate-tree files.

    Single-file cases map to ``{"candidate.py": source}``; package cases map
    every ``*.py`` under the side directory to ``candidate/<relative>``.
    """
    if layout == "single":
        return {"candidate.py": (root / f"{side}.py").read_text(encoding="utf-8")}
    base = root / side
    return {
        f"candidate/{path.relative_to(base).as_posix()}": path.read_text(encoding="utf-8")
        for path in sorted(base.rglob("*.py"))
    }


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing case file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        # A file holding the wrong JSON shape is a content error like invalid
        # JSON above; malformed files raise one exception type.
        raise ValueError(f"{path} must hold a JSON object")  # noqa: TRY004
    return data
