"""Deterministic bytecode-instruction verifier — the objective score for
single-function work.

``measure_ops`` counts the CPython bytecode INSTRUCTIONS a candidate's entry
point executes over a fixed set of held-out input cases, using ``sys.monitoring``
(EVENT_INSTRUCTION) inside a dedicated measurement run. The count is a pure,
deterministic function of (source, cases, interpreter): the same candidate on
the same cases under the same interpreter always produces the same number. The
run is isolated from the host process (an executor run — the Docker sandbox in
production), the interpreter is pinned to ``PYTHONHASHSEED=0`` by the sandbox
executors, and the driver's emission is the same token-framed, fail-closed
pattern the equivalence gate uses: a missing, duplicated, or unparseable ops
frame is an error, never a score.

THE DISCLOSURE (seed learn_0010, carried from Darwin's phase0 FINAL_REPORT
section "Not measured, and why") — READ BEFORE TRUSTING ANY OPS NUMBER:

    Bytecode-instruction counting is deterministic and reproducible but
    STRUCTURALLY BLIND below the bytecode layer. Work performed by C-level
    builtins — str methods, dict operations, itertools, sorted() — does not
    increment Python instruction counters, so a candidate that shifts work
    from Python loops into C builtins scores an "improvement" the counter
    manufactured. Darwin's phase0 largest passing "improvement" was driven by
    exactly this shift (the word_cooccurrence family). Where a large claimed
    gain coincides with replacing Python-level loops with builtin calls,
    cross-check with wall-clock timing on the held-out workload before
    trusting it.

    This disclosure is a contract, not a comment: the code gate attaches it to
    EVERY verdict payload it emits (``GateResult.disclosure``), so no consumer
    can see an ops number without also seeing what the number cannot see.

MEASUREMENT SCOPE: instruction events are enabled per code object
(``set_local_events``) on every code object the candidate module exposes at
import time — module-level functions, class methods, and their nested code
(via ``co_consts`` recursion) — filtered to the candidate's own file. The
measurement loop starts counting after the candidate import completes, so the
score is the candidate's own work on the held-out inputs, not the driver's or
the import's. Code objects created dynamically at call time (exec, closures
constructed inside a call) are not instrumented — the same scope limit as
Darwin's phase0 counter; code-gate equivalence checks bound what such a
candidate can get away with.

TAMPER RESISTANCE (bounded): after the measured calls, the driver verifies the
monitoring hook is still armed (the INSTRUCTION callback is still ours and
every instrumented code object still carries local instruction events) —
a candidate that disarms the counter mid-run yields an error frame, which the
host maps to a fail-closed rejection, not a flattering score. This closes the
cheap disarm; it does NOT make the metric forgery-proof (the candidate and the
driver share one process inside the boundary — see code_gate.py's threat-model
note; the container walls off ACCESS, not co-residence).

OPS COMPARISON CONTRACT (seed learn_0011): baseline and candidate are measured
by THIS SAME module, on the SAME cases, under the SAME executor and
interpreter — a comparison is only meaningful when both sides share the
measurement environment. The never-worse decision itself lives in
``code_gate.compare_to_baseline``.

STDLIB ONLY: this module must stay importable as a bare file inside the
container (the driver needs nothing but the stdlib), so it imports no evogine
package siblings; the executor is duck-typed (any ``run(files, command,
timeout=...)`` returning an object with ``status``/``exit_code``/``stdout``).
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping, Sequence
from typing import Any

# Seed learn_0010: the metric-limit disclosure attached to every verdict
# payload the code gate emits. Single source of truth; do not paraphrase it
# at the call sites.
OPS_DISCLOSURE = (
    "Bytecode-instruction counts are deterministic but blind to C-level "
    "builtin work (str methods, dict ops, itertools): a candidate that shifts "
    "Python loops into C builtins can score an 'improvement' without a real "
    "algorithmic win. Cross-check large gains with wall-clock timing before "
    "trusting them (seed learn_0010)."
)


class VerifierError(RuntimeError):
    """The measurement could not be trusted (fail-closed)."""


class VerifierTimeout(VerifierError):
    """The measurement run hit the sandbox's wall-clock timeout."""


class VerifierCodeError(VerifierError):
    """The MEASURED code raised while being measured (not an environment fault).

    Distinguishing the two matters for the never-worse bar: an incumbent that
    crashes on the replay inputs is exactly the decay the benchmark measures,
    while a broken sandbox or driver must stay fail-closed.
    """


# The trusted in-container measurement driver, same emission pattern as
# verify_case's golden driver (Darwin Phase I1): bind os.write/json.dumps and
# load the inputs BEFORE the untrusted import, keep them as frame-locals inside
# _run(), emit ONE token-framed frame via a raw os.write to a pre-dup'ed fd,
# then os._exit(0) — no atexit/__del__/thread the candidate registered can run
# after us to append or replace the frame. A second frame makes the host fail
# closed.
def _measure_driver(token: str) -> str:
    return (
        "import json, os, sys\n"
        "def _run():\n"
        "    _write, _dumps, _fd = os.write, json.dumps, os.dup(1)\n"
        "    inputs = json.load(open('inputs.json', encoding='utf-8'))\n"
        "    import candidate\n"
        "    fn = getattr(candidate, inputs['entry_point'], None)\n"
        "    if fn is None or not callable(fn):\n"
        "        _frame = {'error': 'missing entry point: ' + inputs['entry_point']}\n"
        "    else:\n"
        "        _target = 'candidate.py'\n"
        "        _mon = sys.monitoring\n"
        "        _ev = _mon.events.INSTRUCTION\n"
        "        def _collect(code, out):\n"
        "            if code not in out:\n"
        "                out.add(code)\n"
        "                for _c in code.co_consts:\n"
        "                    if hasattr(_c, 'co_code'):\n"
        "                        _collect(_c, out)\n"
        "        _codes = set()\n"
        "        def _scan(obj):\n"
        "            code = getattr(obj, '__code__', None)\n"
        "            if code is not None and code.co_filename.endswith(_target):\n"
        "                _collect(code, _codes)\n"
        "            if isinstance(obj, type):\n"
        "                for _v in list(vars(obj).values()):\n"
        "                    _scan(_v)\n"
        "        for _v in list(vars(candidate).values()):\n"
        "            _scan(_v)\n"
        "        _total = 0\n"
        "        def _count(code, instruction_offset):\n"
        "            nonlocal _total\n"
        "            _total += 1\n"
        "        _mon.use_tool_id(0, 'evogine-ops')\n"
        "        _mon.register_callback(0, _ev, _count)\n"
        "        for _c in _codes:\n"
        "            _mon.set_local_events(0, _c, _ev)\n"
        "        _frame = {}\n"
        "        try:\n"
        "            for _c in inputs['cases']:\n"
        "                fn(*_c['args'], **_c['kwargs'])\n"
        "            _prev = _mon.register_callback(0, _ev, _count)\n"
        "            _tampered = (\n"
        "                _prev is not _count\n"
        "                or any(_mon.get_local_events(0, c) != _ev for c in _codes)\n"
        "            )\n"
        "            _frame = ({'ops': _total} if not _tampered\n"
        "                      else {'error': 'monitoring hook disarmed mid-run'})\n"
        "        except BaseException as _e:\n"
        "            _frame = {'error': type(_e).__name__ + ': ' + str(_e)}\n"
        "        finally:\n"
        "            for _c in _codes:\n"
        "                try:\n"
        "                    _mon.set_local_events(0, _c, 0)\n"
        "                except ValueError:\n"
        "                    pass\n"
        "            _mon.free_tool_id(0)\n"
        "    _write(_fd, ('\\n' + " + repr(token) + "\n"
        "               + _dumps(_frame) + '\\n').encode())\n"
        "    os._exit(0)\n"
        "_run()\n"
    )


def _extract_frame(stdout: str, token: str) -> dict[str, Any] | None:
    """Read the driver's ops frame; fail CLOSED unless EXACTLY ONE token-framed
    parseable JSON object is present (same rule as the equivalence drivers):
    zero frames means the candidate suppressed the emission, two or more means
    it injected a forged one alongside the real — either way the measurement is
    unverifiable and rejected."""
    frames = [line[len(token) :] for line in stdout.splitlines() if line.startswith(token)]
    if len(frames) != 1:
        return None
    try:
        frame = json.loads(frames[0])
    except ValueError:
        return None
    return frame if isinstance(frame, dict) else None


def measure_ops(
    source: str,
    entry_point: str,
    cases: Sequence[Mapping[str, Any]],
    executor: Any,
    *,
    timeout: float = 10.0,
) -> int:
    """Count the candidate entry point's bytecode instructions over ``cases``.

    ``cases`` are INPUT-side only mappings with ``args``/``kwargs`` — expected
    outputs are never needed for measurement and never sent to the executor.
    Returns the instruction count; raises :class:`VerifierTimeout` if the run
    hits the wall clock, :class:`VerifierCodeError` when the measured code
    itself raises (a property of the code, not the sandbox), and
    :class:`VerifierError` on any other unverifiable outcome (non-ok run,
    missing/duplicate/unparseable frame, or a disarmed monitoring hook).
    """
    token = secrets.token_hex(16)
    inputs = {
        "entry_point": entry_point,
        "cases": [
            {"args": list(case.get("args", ())), "kwargs": dict(case.get("kwargs", {}))}
            for case in cases
        ],
    }
    result = executor.run(
        {
            "candidate.py": source,
            "inputs.json": json.dumps(inputs),
            "driver.py": _measure_driver(token),
        },
        ["python", "driver.py"],
        timeout=timeout,
    )
    if result.status == "timed_out":
        raise VerifierTimeout("ops measurement timed out")
    if result.status != "ok":
        detail = (result.stdout.strip() or result.stderr.strip() or result.status)[:500]
        raise VerifierError(f"ops measurement run {result.status}: {detail}")
    frame = _extract_frame(result.stdout, token)
    if frame is None:
        raise VerifierError(
            "ops measurement produced no verifiable frame (missing, duplicated, or unparseable)"
        )
    if "error" in frame:
        # The measured code itself raised (e.g. the degraded baseline crashing
        # on a month-end golden input) — a property of the code, not the
        # sandbox; typed so the gate can price the never-worse bar honestly.
        raise VerifierCodeError(f"ops measurement failed: {frame['error']}")
    ops = frame.get("ops")
    if not isinstance(ops, int) or isinstance(ops, bool) or ops < 0:
        raise VerifierError(f"ops frame carries no usable count: {frame!r}")
    return ops
