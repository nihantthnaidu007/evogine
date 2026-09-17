"""The execution boundary every gate check runs behind.

Two executors share one contract: write ``files`` (name -> text) into a fresh
directory, run ``command`` there under a hard wall-clock timeout, and return a
structured :class:`RunResult` with status (``ok`` / ``failed`` / ``crashed`` /
``timed_out``), exit code, capped stdout/stderr, and duration. Staged names are
candidate input: any name that resolves outside the fresh directory is rejected
with :class:`UnsafeCandidatePath` before a single byte is written. Neither
executor knows anything about "tests", "golden I/O", or "equivalence"; that
lives one layer up (``code_gate.py``).

:class:`DockerSandbox` is the real boundary — a port of Darwin's locked-down
container primitive (``sandbox_container.py``, Phase I0). Every candidate is
untrusted AI output, so in production every check MUST run here:

  * NEVER HANG. ``docker run`` is bounded; on expiry the container is
    force-removed in the daemon (``docker rm -f``) and the local client reaped.
    Container output goes to host temp FILES and is read back capped, so even
    an endless ``print`` flood can neither deadlock us nor grow host RAM.
  * ALWAYS CLEAN UP. A ``finally`` always force-removes the container and
    removes the host temp dir and output files, on every path. Every docker
    subcommand is itself time-bounded so a wedged daemon cannot hang teardown.
  * STRUCTURED RESULT. ``RunResult`` mirrors Darwin's ``ContainerResult``
    field-for-field and adds the status taxonomy as an enum.

  THE LOCKDOWN (each flag independently verifiable; the argv test pins it):
  no network (``--network none``); no host filesystem beyond a single
  read-only bind-mount of the per-run dir at ``/sandbox``; memory cap with
  swap disabled; CPU cap; PID cap (contains fork bombs); non-root (``--user
  1000:1000`` + image USER); read-only root filesystem with a writable
  in-RAM tmpfs at ``/tmp`` only; dropped capabilities and
  ``no-new-privileges``; hard wall-clock timeout enforced host-side;
  ``PYTHONHASHSEED=0`` for the deterministic-verifier contract.

:class:`LocalSubprocessSandbox` runs the same contract as a plain local
subprocess. It provides **NO isolation whatsoever** — it exists only for the
test suite (which executes trusted fixture code, where a container would be
pure overhead and CI cannot assume Docker) and for measuring trusted baseline
code locally. Never point it at untrusted code.

PREREQUISITE for :class:`DockerSandbox`: Docker installed and running, and the
sandbox image built once::

    docker build -f docker/Dockerfile.sandbox -t evogine-sandbox:py313 docker/

``DockerSandbox.run`` raises :class:`SandboxUnavailable` with that hint if the
image is missing.
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

IMAGE = "evogine-sandbox:py313"  # built from docker/Dockerfile.sandbox

# Default container caps. Generous enough for a pytest run on the tiny cases,
# tight enough to contain a hostile candidate.
MEMORY = "256m"  # RAM cap; swap is disabled by setting --memory-swap == --memory
CPUS = "1.0"  # at most one core
PIDS = 128  # process cap; contains fork bombs

# Bound the host-side waits so nothing can hang us. _STARTUP_GRACE covers
# image-start latency so a slow-starting legit candidate is not killed early.
_STARTUP_GRACE = 1.0
_OP_TIMEOUT = 15.0
_KILL_GRACE = 5.0

# Cap how much output we read back (per stream), bounding host memory even if
# the candidate floods stdout. The container memory cap stops an in-RAM flood;
# this stops a streaming-print flood.
_MAX_OUTPUT = 1_000_000  # bytes per stream

# OPT-IN defense-in-depth, OFF BY DEFAULT (ported from Darwin decision 0033).
# EVOGINE_SANDBOX_EXTRA_SECURITY_OPTS is a comma-separated list of docker
# --security-opt values (e.g. "seccomp=/abs/seccomp.json"); each becomes an
# ADDITIONAL --security-opt on top of the lockdown — it never removes or alters
# a lockdown flag. Read once at module load: deployment configuration, not a
# per-call argument.
_EXTRA_SECURITY_OPTS_ENV = "EVOGINE_SANDBOX_EXTRA_SECURITY_OPTS"


class RunStatus(str, Enum):
    """The shared four-way outcome taxonomy for every executor."""

    OK = "ok"  # completed successfully (exit 0)
    FAILED = "failed"  # completed with a non-zero exit code
    CRASHED = "crashed"  # terminated abnormally (signal, or killed by the cap)
    TIMED_OUT = "timed_out"  # killed by the sandbox's wall-clock timeout


@dataclass(frozen=True)
class RunResult:
    """What happened during one sandboxed run."""

    status: RunStatus
    exit_code: int | None  # None when timed out
    stdout: str  # capped at _MAX_OUTPUT
    stderr: str  # capped at _MAX_OUTPUT
    duration: float  # wall-clock seconds the run actually took
    timed_out: bool  # convenience flag, == (status is RunStatus.TIMED_OUT)
    oom_killed: bool  # True if the container was killed by the memory cap


class Executor(Protocol):
    """Anything that can run ``command`` against staged ``files`` safely."""

    def run(
        self, files: Mapping[str, str], command: Sequence[str], *, timeout: float
    ) -> RunResult: ...


class SandboxUnavailable(RuntimeError):
    """The Docker boundary cannot be established (missing binary or image)."""


class UnsafeCandidatePath(ValueError):
    """A staged file name escapes (or cannot name a file inside) the staging root.

    Raised by both executors' staging paths. Upstream callers record it as the
    candidate's gate verdict (``swarm.runner._evaluate_reply`` catches it), so
    a traversal attempt fails the case instead of crashing the loop.
    """


def _load_extra_security_opts() -> tuple[str, ...]:
    raw = os.environ.get(_EXTRA_SECURITY_OPTS_ENV, "")
    return tuple(opt.strip() for opt in raw.split(",") if opt.strip())


EXTRA_SECURITY_OPTS = _load_extra_security_opts()


def _staged_path(workdir: str, fname: str) -> str:
    """Resolve ``fname`` under the staging root and refuse any escape.

    The ``files`` keys are candidate-supplied, so a name is attacker input:
    joined naively, ``candidate/../../x`` or an absolute path would land
    OUTSIDE the staging dir — a host write before any container exists (C1,
    v1 security review). This mirrors the resolve-under-root check in
    ``bench/gate.py``'s ``candidate_import_context``. The staging root is a
    fresh ``mkdtemp``, so there is no pre-existing symlink for ``resolve()``
    to travel through. Resolving to the root itself ("" / "." / "a/..") is
    also rejected: such a name can never open a file and would otherwise
    surface downstream as a raw ``IsADirectoryError``.
    """
    if "\x00" in fname:
        raise UnsafeCandidatePath(f"staged file name contains a null byte: {fname!r}")
    root = Path(workdir).resolve()
    resolved = (root / fname).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise UnsafeCandidatePath(f"staged file name escapes the staging root: {fname!r}")
    return str(resolved)


def _stage_files(workdir: str, files: Mapping[str, str]) -> None:
    """Write ``files`` (name -> text) under ``workdir``, creating parent dirs."""
    for fname, content in files.items():
        path = _staged_path(workdir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


class DockerSandbox:
    """The locked-down container executor. The default production boundary."""

    def run(
        self,
        files: Mapping[str, str],
        command: Sequence[str],
        *,
        timeout: float = 10.0,
        image: str = IMAGE,
        memory: str = MEMORY,
        cpus: str = CPUS,
        pids: int = PIDS,
    ) -> RunResult:
        """Run ``command`` in a locked-down container with ``files`` read-only
        mounted at /sandbox. Enforces a hard ``timeout`` in seconds: on expiry
        the container is force-removed and the result is marked timed out. The
        call always returns within ~timeout + a bounded grace, and the container
        and host temp artifacts are always removed before returning."""
        self._require_image(image)
        name = "evogine_sbx_" + uuid.uuid4().hex  # unique; the teardown handle
        # Output goes to host files OUTSIDE the mounted dir (the container never
        # sees them), so a full pipe can never deadlock us and we read back only
        # a capped amount.
        out_fd, out_path = tempfile.mkstemp(prefix="evogine_sbx_out_")
        err_fd, err_path = tempfile.mkstemp(prefix="evogine_sbx_err_")
        os.close(out_fd)
        os.close(err_fd)
        workdir: str | None = None
        try:
            workdir = self._stage(files)
            argv = _run_argv(name, workdir, command, image, memory, cpus, pids)
            start = time.monotonic()
            with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
                # No --rm: keep the container after exit so we can inspect
                # OOM/exit; the finally below force-removes it on every path.
                proc = subprocess.Popen(argv, stdout=out_f, stderr=err_f, start_new_session=True)
                timed_out = False
                try:
                    proc.communicate(timeout=timeout + _STARTUP_GRACE)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    # Authoritatively stop the runaway IN THE DAEMON (killing
                    # the local `docker run` client would not stop it).
                    _docker(["rm", "-f", name])
                    try:
                        proc.wait(timeout=_KILL_GRACE)
                    except subprocess.TimeoutExpired:
                        proc.kill()  # reap the docker client itself; never block
                        try:
                            proc.wait(timeout=_KILL_GRACE)
                        except subprocess.TimeoutExpired:
                            pass
            duration = time.monotonic() - start
            stdout = _read_capped(out_path, _MAX_OUTPUT)
            stderr = _read_capped(err_path, _MAX_OUTPUT)

            if timed_out:
                return RunResult(RunStatus.TIMED_OUT, None, stdout, stderr, duration, True, False)

            # Not timed out: the container exists (no --rm), so inspect is
            # authoritative for the exit code and the OOM flag.
            exit_code, oom = _inspect(name)
            if exit_code is None:
                exit_code = proc.returncode
            return RunResult(
                _classify(exit_code, oom), exit_code, stdout, stderr, duration, False, oom
            )
        finally:
            # Guaranteed teardown on every path (success, failure, crash, timeout).
            _docker(["rm", "-f", name])
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)
            for p in (out_path, err_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    @staticmethod
    def _stage(files: Mapping[str, str]) -> str:
        """Write ``files`` into a fresh host temp dir and return its path.

        mkdtemp creates the dir mode 0o700 by design. On a Linux Docker host a
        read-only bind mount preserves the host directory's owner and mode, so
        the container's uid 1000 (a different uid than ours) could not traverse
        or read a 0o700 dir we own. We widen the staged tree — only ADDING
        non-owner read/traverse bits, never narrowing the owner's — because the
        staged files are the candidate's own code, not secret, and the temp dir
        is removed by run()'s finally block. macOS Docker Desktop ignores host
        uid/mode, so the widening is harmless there.
        """
        workdir = tempfile.mkdtemp(prefix="evogine_sbx_")
        _stage_files(workdir, files)
        for root, _dirs, fnames in os.walk(workdir):
            os.chmod(root, stat.S_IMODE(os.stat(root).st_mode) | 0o055)
            for fname in fnames:
                fpath = os.path.join(root, fname)
                os.chmod(fpath, stat.S_IMODE(os.stat(fpath).st_mode) | 0o044)
        return workdir

    @staticmethod
    def _require_image(image: str) -> None:
        """Fail fast with a build hint if the sandbox image is missing, instead
        of letting ``docker run`` return an opaque 125 later.

        inspect-by-tag is the primary check; under containerd image storage it
        can false-negative, so a non-zero inspect does NOT abort: we fall back
        to listing local images (a different resolution path) before raising."""
        rc, _, _ = _docker(["image", "inspect", image])
        if rc == 0:
            return  # present by tag: happy path, no extra call
        rc_ls, out, _ = _docker(["image", "ls", "--format", "{{.Repository}}:{{.Tag}}"])
        if rc_ls == 0 and image in out.splitlines():
            return  # inspect false-negatived; image is present
        raise SandboxUnavailable(
            f"sandbox image {image!r} not found. Build it once with:\n"
            f"    docker build -f docker/Dockerfile.sandbox -t {image} docker/"
        )


def _extra_security_opt_args(extra_security_opts: Sequence[str]) -> list[str]:
    """Expand opt-in extras into --security-opt argv pairs. Empty (the default)
    expands to nothing, so the produced argv is byte-identical to the base
    lockdown; each entry only ADDS restriction."""
    args: list[str] = []
    for opt in extra_security_opts:
        args += ["--security-opt", opt]
    return args


def _run_argv(
    name: str,
    workdir: str,
    command: Sequence[str],
    image: str,
    memory: str,
    cpus: str,
    pids: int,
    extra_security_opts: Sequence[str] | None = None,
) -> list[str]:
    """The locked-down ``docker run`` argv. Every flag here is one lockdown item.

    extra_security_opts is the OPT-IN hook: None (the default) uses the
    module-level EXTRA_SECURITY_OPTS (empty unless the operator set the env
    var), read at CALL time so operator env vars and tests both take effect.
    """
    opts = EXTRA_SECURITY_OPTS if extra_security_opts is None else extra_security_opts
    return [
        "docker",
        "run",
        "--name",
        name,
        "--network",
        "none",  # no network
        "--memory",
        memory,
        "--memory-swap",
        memory,  # == memory -> swap disabled -> real OOM
        "--cpus",
        str(cpus),  # CPU cap
        "--pids-limit",
        str(pids),  # process cap -> contains fork bombs
        "--read-only",  # read-only root filesystem
        "--tmpfs",
        "/tmp:rw,size=64m,mode=1777",  # only writable area, in RAM
        "--cap-drop",
        "ALL",  # drop all Linux capabilities
        "--security-opt",
        "no-new-privileges",
        # OPT-IN extras (off by default): zero elements when unset.
        *_extra_security_opt_args(opts),
        "--user",
        "1000:1000",  # non-root (image USER is also sandbox)
        "-v",
        f"{workdir}:/sandbox:ro",  # the ONLY host path, read-only
        "-w",
        "/sandbox",
        "-e",
        "HOME=/tmp",  # writable HOME on the read-only root
        "-e",
        "TMPDIR=/tmp",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp",
        "-e",
        "PYTHONHASHSEED=0",  # deterministic-verifier contract: hash-stable runs
        image,
        *command,
    ]


def _classify(exit_code: int | None, oom: bool) -> RunStatus:
    """Map a finished container's exit code to the shared taxonomy. OOM is an
    abnormal kill -> CRASHED. Docker reports a signal death as 128+signal, so
    exit codes >= 128 are signal deaths -> CRASHED; 0 is OK; any other non-zero
    exit (assertion=1, exception=1, ModuleNotFound=1, and docker infra errors
    125/126/127) is FAILED, with docker's stderr captured so infra errors show."""
    if oom:
        return RunStatus.CRASHED
    if exit_code == 0:
        return RunStatus.OK
    if exit_code is not None and 0 < exit_code < 128:
        return RunStatus.FAILED
    # Version-robust: an OOM kill is SIGKILL -> exit 137 (>= 128), landing here
    # even when the cgroup-dependent .State.OOMKilled flag is unset.
    return RunStatus.CRASHED


def _inspect(name: str) -> tuple[int | None, bool]:
    """Return (exit_code, oom_killed) from ``docker inspect``, or (None, False)
    if it cannot be read. Bounded; never hangs the caller."""
    rc, out, _ = _docker(["inspect", "--format", "{{.State.ExitCode}} {{.State.OOMKilled}}", name])
    if rc != 0:
        return None, False
    parts = out.split()
    if len(parts) != 2 or not parts[0].lstrip("-").isdigit():
        return None, False
    return int(parts[0]), parts[1] == "true"


def _docker(args: Sequence[str]) -> tuple[int, str, str]:
    """Run a ``docker`` subcommand with a hard timeout. Returns (rc, stdout,
    stderr); on timeout or launch failure returns non-zero rc and empty output."""
    try:
        p = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=_OP_TIMEOUT,
            check=False,  # return codes are inspected, not raised
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "docker timed out"
    except OSError as e:
        return 125, "", str(e)


def _read_capped(path: str, limit: int) -> str:
    """Read up to ``limit`` bytes from ``path``, appending a marker if larger."""
    try:
        with open(path, "rb") as f:
            data = f.read(limit + 1)
    except OSError:
        return ""
    if len(data) > limit:
        return data[:limit].decode("utf-8", "replace") + f"\n...[truncated at {limit} bytes]"
    return data.decode("utf-8", "replace")


# Environment variables passed through to the local candidate subprocess —
# everything else is dropped (H1, v1 security review).
_LOCAL_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")


def _local_env() -> dict[str, str]:
    """Build a minimal environment for the local candidate subprocess.

    The local executor is not a security boundary, but it still must not hand
    the candidate the loop process's secrets: a naive ``{**os.environ, ...}``
    inherited everything, including ``ANTHROPIC_API_KEY`` during a live pilot
    (H1, v1 security review). Pass through only what a CPython subprocess
    needs to run — path lookups, home, locale, temp dir — plus the determinism
    contract the Docker executor's container env already pins.
    """
    env = {name: os.environ[name] for name in _LOCAL_ENV_ALLOWLIST if name in os.environ}
    env["PYTHONHASHSEED"] = "0"  # same determinism contract as the container
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class LocalSubprocessSandbox:
    """Plain local subprocess executor — **NOT a security boundary**.

    Provides the same RunResult contract and timeout discipline as
    DockerSandbox so the gate logic is executor-agnostic, but the command runs
    as an ordinary local process with full host access. Only run TRUSTED code
    here (test fixtures, trusted baseline measurement); never untrusted
    candidate code in production — use DockerSandbox.

    A command spelled ``python ...`` is rewritten to ``sys.executable`` so the
    same command list works against both executors (the container provides its
    own ``python``; the host venv's is the only correct one locally).

    The candidate subprocess receives a minimal allowlisted environment
    (``_local_env()``: PATH, HOME, LANG, LC_ALL, TMPDIR plus the determinism
    pins) — never the loop process's full environ, which carries secrets such
    as ``ANTHROPIC_API_KEY`` during a live pilot (H1, v1 security review).
    """

    def run(
        self,
        files: Mapping[str, str],
        command: Sequence[str],
        *,
        timeout: float = 10.0,
    ) -> RunResult:
        if command and command[0] == "python":
            command = (sys.executable, *command[1:])
        workdir = tempfile.mkdtemp(prefix="evogine_local_sbx_")
        try:
            _stage_files(workdir, files)
            env = _local_env()
            start = time.monotonic()
            proc = subprocess.Popen(
                command,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,  # own process group, so teardown can kill it all
            )
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the whole group: a breakaway grandchild must not hold our
                # pipes or outlive the timeout.
                os.killpg(proc.pid, signal.SIGKILL)
                out, err = proc.communicate()
                duration = time.monotonic() - start
                return RunResult(
                    RunStatus.TIMED_OUT,
                    None,
                    _capped(out),
                    _capped(err),
                    duration,
                    True,
                    False,
                )
            duration = time.monotonic() - start
            exit_code = proc.returncode
            # Popen reports signal deaths negative (-N); Docker reports them
            # 128+N. Both are abnormal terminations -> CRASHED.
            if exit_code == 0:
                status = RunStatus.OK
            elif exit_code is not None and (exit_code < 0 or exit_code >= 128):
                status = RunStatus.CRASHED
            else:
                status = RunStatus.FAILED
            return RunResult(status, exit_code, _capped(out), _capped(err), duration, False, False)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _capped(data: bytes, limit: int = _MAX_OUTPUT) -> str:
    if len(data) > limit:
        return data[:limit].decode("utf-8", "replace") + f"\n...[truncated at {limit} bytes]"
    return data.decode("utf-8", "replace")
