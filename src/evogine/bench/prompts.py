"""Builder prompt templates for candidate generation.

Template text is a leak surface: no expected value from any case's
properties.json may appear here. tests/test_oracle_leak.py scans every string
constant in this module (and every scenario.md) against the union of all
cases' oracle values. The templates carry structure only; task-specific
content is injected at build time from generator-visible case material
(meta description, scenario.md, degraded source) and, for the warm arm,
promoted engine records.
"""

from __future__ import annotations

from evogine.bench.task import BenchTask
from evogine.engine.records import LearningRecord

BUILDER_SYSTEM_PROMPT = (
    "You are the repair builder in a decay-pressure benchmark. You receive the "
    "current source of a small Python module plus a pressure scenario. Repair "
    "the code so it keeps its current, correct behavior and also holds up "
    "under that scenario's pressure. Keep the public contract described in the "
    "task. Return the complete repaired source and nothing else."
)

BUILDER_PACKAGE_NOTE = (
    " This task is multi-file: reply with a single JSON object exactly of the"
    ' form {"files": {"<relative path>": "<full source>"}} where relative paths'
    ' are package-root relative (for example "__init__.py", "queue.py").'
    " No other text."
)

BUILDER_TASK_TEMPLATE = (
    "Task: repair the module {case_id} (decay axis: {decay_axis}).\n"
    "\n"
    "Task description:\n{description}\n"
    "\n"
    "Pressure scenario:\n{scenario}\n"
    "\n"
    "Current source to repair:\n{source}\n"
    "\n"
    "{knowledge}Reproduce the full module source now."
)


def render_knowledge(records: list[LearningRecord]) -> str:
    """Render promoted records as a knowledge block; empty for the cold arm."""
    if not records:
        return ""
    lines = [
        f"- [{record.type}] {record.title}: {record.body.fix_strategy}" for record in records
    ]
    return "Promoted knowledge to apply:\n" + "\n".join(lines) + "\n\n"


def build_builder_messages(
    task: BenchTask, records: list[LearningRecord]
) -> list[dict[str, str]]:
    """Compose the builder chat for one candidate generation.

    ``records`` is the retrieval result: promoted records for the warm arm,
    an empty list for the cold arm (retrieval suppressed via NullEngine).
    Hidden properties are never consulted here — only generator-visible case
    material travels in the messages.
    """
    system = BUILDER_SYSTEM_PROMPT + (BUILDER_PACKAGE_NOTE if task.layout == "package" else "")
    source_blob = "\n".join(
        f"### {relative}\n{source}" for relative, source in sorted(task.baseline_files.items())
    )
    user = BUILDER_TASK_TEMPLATE.format(
        case_id=task.case_id,
        decay_axis=task.decay_axis,
        description=task.description,
        scenario=task.scenario_brief,
        source=source_blob,
        knowledge=render_knowledge(records),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
