"""Differential equivalence at the public API — the behavioral half of the gate
and the objective score for multi-file work (ported from Darwin's mdiffer.py +
mverify.py).

Semantics (unchanged from the M0 spike both Darwin modules promote): a unit is
a black box exporting its public class, and a single function is a black box
exporting its entry point; for every sequence of API calls (or every golden
I/O case) the candidate must produce identical per-call observables — return
value, or exception type name + message — compared under STRICT TYPED equality
(:func:`canon`): ``True`` is not ``1``, list/tuple order is observable, dict
key order is not. Private internals are never inspected.

Information flow (the container is the boundary): the executor receives only
the CANDIDATE source, the INPUT side of the cases (the stdlib-only corpus
generator for multi-file work; raw args/kwargs for single-function work), and
this module (the replay machinery, which contains no case data). The expected
outputs, the original source, and the comparison NEVER enter the executor.
The candidate-side driver replays against the candidate and emits one
token-framed frame of per-call canonical outcomes; the HOST replays the
original (or holds the golden outputs) and compares in corpus order; the first
divergence decides.

ORACLE-FREE FEEDBACK (a deliberate hardening over Darwin): a divergence
witness here carries the CANDIDATE's outcome and where it diverged — but NOT
the original's outcome. Darwin's mdiffer witness included both sides and
mloop stripped the original's lines before the generator saw them
(``oracle_markers.py``); there is no downstream sanitizer to trust in evogine,
so the gate's own payload simply never contains the oracle side. The original
outcome is withheld from feedback exactly like the golden files are withheld
from the container: a builder agent that could read it could hardcode it.

FAIL CLOSED: a candidate that cannot be loaded or replayed (import error,
missing public method), or whose result frame is missing, duplicated,
unparseable, or mismatched against the batch, is NOT equivalent — surfaced as
a :class:`ReplayError` (timeout and crash carry their own subclasses so the
code gate can map them to the right verdict).

THE FORGERY BOUNDARY (carried verbatim from Darwin v2.0 / Phase I1 —
documented, NOT re-solved): the candidate and the driver share ONE process
inside the container. The boundary closes ACCESS (no host filesystem, no
network, no oracle to read) but does not split the process, so a candidate can
still forge the outcomes it emits in-process (the PEP 667 frame-locals
serializer hijack, or compute-and-lie). The token-framed single-frame
extraction raises the bar for naive cheats; it is NOT forgery-proof and no
in-process scheme can be. See code_gate.py and Darwin's SECURITY.md.

STDLIB ONLY: this module is copied INTO the container as the driver's replay
machinery (staged as ``equivalence.py``), so it imports no evogine package
siblings; the executor is duck-typed (any ``run(files, command, timeout=...)``
returning an object with ``status``/``stdout``).
"""

from __future__ import annotations

import importlib.util
import json
import secrets
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ errors


class ReplayError(RuntimeError):
    """The candidate could not be verified as equivalent (fail-closed)."""


class ReplayTimeout(ReplayError):
    """The candidate replay hit the sandbox's wall-clock timeout."""


class ReplayCrash(ReplayError):
    """The candidate replay died abnormally (signal, or a resource cap)."""


# ----------------------------------------------------------------- loading


def load_unit(pkg_dir: str | Path):
    """Import a unit directory as a uniquely named package.

    Unique per call with NO shared counter: a uuid is collision-free without
    coordination, so concurrent loads cannot collide on a sys.modules name
    (Darwin decision 0024). The unit directory goes on sys.path only for the
    duration of the import (intra-unit imports are absolute top-level names,
    matching how the in-container driver loads ``candidate/``), and the unit's
    own modules are purged from sys.modules afterwards so a later unit that
    reuses the same module names re-imports fresh instead of inheriting cached
    code."""
    alias = f"evogine_unit_{uuid.uuid4().hex}"
    root = Path(pkg_dir).resolve()
    init = root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        alias, init, submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for the unit at {init}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(root))
        stale = [
            name
            for name, m in list(sys.modules.items())
            if name != alias
            and getattr(m, "__file__", None) is not None
            and str(root) in str(m.__file__)
        ]
        for name in stale:
            del sys.modules[name]
    return mod


def _load_unit_from_files(files: Mapping[str, str]):
    """Materialize a trusted source tree and load it as a unit package.

    Used for the BASELINE, which is trusted host-side code — never for the
    candidate, which must never run in this process."""
    if "__init__.py" not in files:
        raise ValueError("baseline unit is missing __init__.py")
    with tempfile.TemporaryDirectory(prefix="evogine_unit_") as tmp:
        for name, content in files.items():
            path = Path(tmp) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return load_unit(tmp)


def _load_corpus_source(corpus_source: str) -> list[tuple[str, tuple, tuple]]:
    """Load a stdlib-only corpus module (defining ``build_corpus()``) from its
    source text. Uniquely aliased per call (decision 0024, as load_unit)."""
    name = f"evogine_corpus_{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="evogine_corpus_") as tmp:
        path = Path(tmp) / "corpus.py"
        path.write_text(corpus_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build an import spec for the corpus at {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        sequences = mod.build_corpus()
    if not isinstance(sequences, list) or not all(
        isinstance(s, tuple) and len(s) == 3 for s in sequences
    ):
        raise ValueError("corpus build_corpus() must return (name, ctor_args, calls) triples")
    return sequences


# --------------------------------------------------------------- comparison


def canon(v: Any) -> tuple:
    """Canonical form for strict typed comparison of observable values.

    bool and int canonicalize differently (``True != 1`` here, unlike ``==``).
    Lists/tuples stay ordered; dicts compare order-insensitively."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        return ("float", repr(v))
    if isinstance(v, str):
        return ("str", v)
    if v is None:
        return ("none",)
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(canon(x) for x in v))
    if isinstance(v, dict):
        return ("dict", tuple(sorted((canon(k), canon(val)) for k, val in v.items())))
    return ("other", type(v).__name__, repr(v))


def canon_outcome(outcome: tuple) -> tuple:
    if outcome[0] == "ok":
        return ("ok", canon(outcome[1]))
    return outcome  # ("raise", type_name, message) is already canonical


def _canon_norm(method: str, outcome: tuple) -> list:
    """A canonical outcome, JSON-round-tripped so host-side tuples compare
    equal to the candidate's JSON-decoded lists."""
    return json.loads(json.dumps([method, canon_outcome(outcome)]))


def _outcome(fn, args, kwargs) -> tuple:
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001 — capturing ANY failure as the observable outcome is the point
        return ("raise", type(e).__name__, str(e))


def run_sequence(unit_cls, ctor_args, calls) -> list[tuple]:
    """Run one sequence on a fresh instance; return per-call outcomes."""
    outcomes = []
    ctor = _outcome(unit_cls, tuple(ctor_args), {})
    outcomes.append(("__init__", ctor))
    if ctor[0] == "raise":
        return outcomes  # no instance to continue with
    instance = ctor[1]
    outcomes[0] = ("__init__", ("ok", "<instance>"))
    for call in calls:
        method, args = call[0], call[1]
        kwargs = call[2] if len(call) > 2 else {}
        outcomes.append((method, _outcome(getattr(instance, method), args, kwargs)))
    return outcomes


# ----------------------------------------------------- driver machinery

# Reference-sized canonical payload allowed per executor run. The hard limit is
# the sandbox's 1 MB stdout read-back; 250 KB of reference payload leaves ~4x
# headroom for a divergent candidate's larger emission. A frame that still
# overflows the cap fails closed below.
_BATCH_BUDGET = 250_000

# Padding for length-mismatched outcome lists, in the JSON-normalized form the
# comparison operates on.
_ABSENT = ["<absent>", ["<absent>"]]


def _self_source() -> str:
    """This module's own source, for staging the replay machinery into the
    container. The module is stdlib-only, so the copy is self-contained."""
    return Path(__file__).read_text(encoding="utf-8")


def _extract_frame(stdout: str, token: str) -> dict[str, Any] | None:
    """Read the driver's result frame; fail CLOSED unless EXACTLY ONE
    token-prefixed parseable frame is present: zero means the candidate
    suppressed/never wrote it, two-or-more means it injected a forged frame
    ALONGSIDE the real one — either way the replay is unverifiable."""
    frames = [line[len(token) :] for line in stdout.splitlines() if line.startswith(token)]
    if len(frames) != 1:
        return None
    try:
        frame = json.loads(frames[0])
    except ValueError:
        return None
    return frame if isinstance(frame, dict) else None


def _classify_replay(res) -> None:
    """Raise the mapped ReplayError for a non-ok executor result."""
    if res.status == "timed_out":
        raise ReplayTimeout("candidate replay timed out")
    if res.status == "crashed":
        raise ReplayCrash(f"candidate replay crashed: {_detail(res)}")
    raise ReplayError(f"candidate replay failed: {_detail(res)}")


def _detail(res) -> str:
    payload = res.stdout.strip() or res.stderr.strip() or res.status
    return str(payload)[:500]


def _witness(sequence: str, call_index: int, call: str, candidate_outcome: tuple) -> dict:
    """A divergence witness WITHOUT the oracle side (see module docstring)."""
    return {
        "sequence": sequence,
        "call_index": call_index,
        "call": call,
        "candidate": json.loads(json.dumps(candidate_outcome)),
        "original": "<withheld: oracle material never leaves the host>",
    }


# --------------------------------------------- single-function golden replay


def _function_driver(token: str) -> str:
    """Trusted in-container driver: call the entry point over the inputs and
    emit the actual results. Same emission pattern as Darwin's
    verify_case._golden_driver: pre-bound write/dumps taken before the
    untrusted import, one token-framed raw-fd write, then os._exit(0)."""
    return (
        "import json, os\n"
        "def _run():\n"
        "    _write, _dumps, _fd = os.write, json.dumps, os.dup(1)\n"
        "    inputs = json.load(open('inputs.json', encoding='utf-8'))\n"
        "    import candidate\n"
        "    fn = getattr(candidate, inputs['entry_point'])\n"
        "    results = [fn(*c['args'], **c['kwargs']) for c in inputs['cases']]\n"
        "    _write(_fd, ('\\n' + " + repr(token) + "\n"
        "               + _dumps({'results': results}) + '\\n').encode())\n"
        "    os._exit(0)\n"
        "_run()\n"
    )


def check_function_equivalence(
    source: str,
    entry_point: str,
    cases: Sequence[Mapping[str, Any]],
    executor: Any,
    *,
    timeout: float = 10.0,
) -> tuple[bool, dict | None, dict]:
    """Single-function differential replay against golden I/O.

    ``cases`` carry ``args``/``kwargs`` (the INPUT side, which goes to the
    executor) and ``output`` (the ORACLE, which never does). Returns
    ``(equivalent, witness, stats)``; raises the ReplayError family when the
    replay itself cannot be verified."""
    token = secrets.token_hex(16)
    inputs = {
        "entry_point": entry_point,
        "cases": [
            {"args": list(case.get("args", ())), "kwargs": dict(case.get("kwargs", {}))}
            for case in cases
        ],
    }
    res = executor.run(
        {
            "candidate.py": source,
            "inputs.json": json.dumps(inputs),
            "driver.py": _function_driver(token),
        },
        ["python", "driver.py"],
        timeout=timeout,
    )
    if res.status != "ok":
        _classify_replay(res)
    actuals = _extract_frame(res.stdout, token)
    stats = {"cases": len(cases), "executor_runs": 1}
    if (
        actuals is None
        or not isinstance(actuals.get("results"), list)
        or len(actuals["results"]) != len(cases)
    ):
        raise ReplayError(
            "candidate replay produced no verifiable results "
            "(missing, duplicated, or unparseable frame)"
        )
    for i, (case, actual) in enumerate(zip(cases, actuals["results"])):
        # Strict typed equality on both sides (canon closes the True == 1 edge
        # that Darwin v1's raw JSON comparison inherited).
        expected = canon(json.loads(json.dumps(case["output"])))
        got = canon(actual)
        if got != expected:
            call = f"{entry_point}(*{case.get('args', ())!r}, **{case.get('kwargs', {})!r})"
            return False, _witness("golden", i, call, got), stats
    return True, None, stats


# ------------------------------------------------ multi-file unit equivalence


def _unit_driver(entry: str, start: int, stop: int, token: str) -> str:
    """Trusted in-container replay driver for one batch span of sequences
    (mverify's _unit_driver, importing this module as ``equivalence``)."""
    return (
        "import json, os\n"
        "def _run():\n"
        "    _write, _dumps, _fd = os.write, json.dumps, os.dup(1)\n"
        "    import equivalence, corpus\n"
        f"    batch = corpus.build_corpus()[{start}:{stop}]\n"
        "    try:\n"
        f"        cls = getattr(equivalence.load_unit('candidate'), {entry!r})\n"
        "        payload = [[name, [[m, equivalence.canon_outcome(o)]\n"
        "                           for m, o in equivalence.run_sequence(cls, ctor, calls)]]\n"
        "                   for name, ctor, calls in batch]\n"
        "        frame = {'sequences': payload}\n"
        "    except BaseException as e:\n"
        "        frame = {'error': type(e).__name__ + ': ' + str(e)}\n"
        # Compact separators keep the emission the size _batch_spans budgeted.
        "    _write(_fd, ('\\n' + " + repr(token) + "\n"
        "               + _dumps(frame, separators=(',', ':')) + '\\n').encode())\n"
        "    os._exit(0)\n"
        "_run()\n"
    )


def _batch_spans(reference: list[list]) -> list[tuple[int, int]]:
    """Contiguous (start, stop) sequence spans whose reference payload stays
    under _BATCH_BUDGET bytes (a lone oversized sequence gets its own span)."""
    spans = []
    start, size = 0, 0
    for i, seq in enumerate(reference):
        seq_size = len(json.dumps(seq, separators=(",", ":")))
        if i > start and size + seq_size > _BATCH_BUDGET:
            spans.append((start, i))
            start, size = i, 0
        size += seq_size
    spans.append((start, len(reference)))
    return spans


def check_unit_equivalence(
    baseline_files: Mapping[str, str],
    candidate_files: Mapping[str, str],
    entry_class: str,
    corpus_source: str,
    executor: Any,
    *,
    timeout: float = 30.0,
) -> tuple[bool, dict | None, dict]:
    """Multi-file differential equivalence with the candidate inside the
    executor and the original replayed host-side (mverify's pattern).

    Returns ``(equivalent, witness, stats)`` in mdiffer's shape; raises the
    ReplayError family when the candidate replay cannot be verified."""
    if "__init__.py" not in baseline_files:
        raise ValueError("baseline unit is missing __init__.py")
    sequences = _load_corpus_source(corpus_source)

    # The trusted original replays HERE in this process, unchanged (mdiffer).
    # A baseline missing the entry class is fail-closed like any other
    # unverifiable comparison, not a host crash.
    orig_cls = getattr(_load_unit_from_files(baseline_files), entry_class, None)
    if not callable(orig_cls):
        raise ReplayError(f"baseline unit has no entry class: {entry_class}")
    reference = [
        [_canon_norm(m, o) for m, o in run_sequence(orig_cls, ctor, calls)]
        for _name, ctor, calls in sequences
    ]

    base_files: dict[str, str] = {
        **{f"candidate/{name}": content for name, content in candidate_files.items()},
        "equivalence.py": _self_source(),
        "corpus.py": corpus_source,
    }

    calls_compared = 0
    runs = 0
    for span_start, span_stop in _batch_spans(reference):
        token = secrets.token_hex(16)
        files = {
            **base_files,
            "driver.py": _unit_driver(entry_class, span_start, span_stop, token),
        }
        res = executor.run(files, ["python", "driver.py"], timeout=timeout)
        runs += 1

        if res.status != "ok":
            _classify_replay(res)
        frame = _extract_frame(res.stdout, token)
        if frame is None:
            raise ReplayError(
                "candidate replay produced no verifiable result frame "
                "(missing, duplicated, or unparseable)"
            )
        if "error" in frame:
            raise ReplayError(f"candidate failed to replay: {frame['error']}")
        emitted = frame.get("sequences")
        if not (
            isinstance(emitted, list)
            and len(emitted) == span_stop - span_start
            and all(isinstance(e, list) and len(e) == 2 and isinstance(e[1], list) for e in emitted)
            and [e[0] for e in emitted] == [s[0] for s in sequences[span_start:span_stop]]
        ):
            raise ReplayError("candidate replay frame does not match the batch")

        for offset, seq_idx in enumerate(range(span_start, span_stop)):
            name, _ctor, calls = sequences[seq_idx]
            ref_calls = reference[seq_idx]
            cand_calls = emitted[offset][1]
            for idx in range(max(len(ref_calls), len(cand_calls))):
                oa = ref_calls[idx] if idx < len(ref_calls) else _ABSENT
                ob = cand_calls[idx] if idx < len(cand_calls) else _ABSENT
                calls_compared += 1
                if oa != ob:
                    call = "__init__" if idx == 0 else repr(calls[idx - 1])
                    witness = _witness(name, idx, call, tuple(ob[1]))
                    return (
                        False,
                        witness,
                        {
                            "calls_compared": calls_compared,
                            "sequences": len(sequences),
                            "executor_runs": runs,
                        },
                    )
    return (
        True,
        None,
        {
            "calls_compared": calls_compared,
            "sequences": len(sequences),
            "executor_runs": runs,
        },
    )
