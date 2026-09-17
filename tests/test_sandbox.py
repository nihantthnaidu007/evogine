"""Sandbox executor tests: the local subprocess contract (no Docker needed),
the pinned lockdown argv (ported tier-a from Darwin's test_sandbox_argv.py),
and Docker-gated smoke tests that skip when the boundary is unavailable."""

import signal

import pytest

from evogine.gates import sandbox as sbx

LOCAL = sbx.LocalSubprocessSandbox()

# Fixed inputs for the argv tests (Darwin's pins, renamed).
NAME = "evogine_sbx_unit"
WORKDIR = "/host/run"
CMD = ["python", "candidate.py"]

# The EXACT current lockdown argv, pinned byte-for-byte. Every flag here is
# load-bearing; this list is the contract the opt-in must not disturb.
EXPECTED_DEFAULT_ARGV = [
    "docker",
    "run",
    "--name",
    NAME,
    "--network",
    "none",
    "--memory",
    "256m",
    "--memory-swap",
    "256m",
    "--cpus",
    "1.0",
    "--pids-limit",
    "128",
    "--read-only",
    "--tmpfs",
    "/tmp:rw,size=64m,mode=1777",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--user",
    "1000:1000",
    "-v",
    "/host/run:/sandbox:ro",
    "-w",
    "/sandbox",
    "-e",
    "HOME=/tmp",
    "-e",
    "TMPDIR=/tmp",
    "-e",
    "PYTHONDONTWRITEBYTECODE=1",
    "-e",
    "PYTHONPYCACHEPREFIX=/tmp",
    "-e",
    "PYTHONHASHSEED=0",
    sbx.IMAGE,
    "python",
    "candidate.py",
]


def _argv(extra_security_opts=()):
    return sbx._run_argv(
        NAME,
        WORKDIR,
        CMD,
        sbx.IMAGE,
        sbx.MEMORY,
        sbx.CPUS,
        sbx.PIDS,
        extra_security_opts=extra_security_opts,
    )


# =========================================================== tier a: argv


def test_default_argv_is_byte_identical():
    """With no opt-in, the produced argv equals the lockdown argv exactly."""
    assert _argv() == EXPECTED_DEFAULT_ARGV


def test_opt_in_only_appends_security_opts():
    """Removing the opt-in's --security-opt pairs must restore the default
    argv exactly: the hook appends, it never removes or alters a flag."""
    argv = _argv(("seccomp=/abs/seccomp.json", "apparmor=profile"))
    pruned = list(argv)
    for opt in ("seccomp=/abs/seccomp.json", "apparmor=profile"):
        i = pruned.index(opt)
        assert pruned[i - 1] == "--security-opt"
        del pruned[i - 1 : i + 1]
    assert pruned == EXPECTED_DEFAULT_ARGV


def test_extra_security_opts_env_var(monkeypatch):
    """The module-level opt-in parses the documented comma-separated format."""
    monkeypatch.setenv(sbx._EXTRA_SECURITY_OPTS_ENV, "seccomp=/x.json, apparmor=y")
    assert sbx._load_extra_security_opts() == ("seccomp=/x.json", "apparmor=y")
    monkeypatch.delenv(sbx._EXTRA_SECURITY_OPTS_ENV)
    assert sbx._load_extra_security_opts() == ()


# ============================================== tier a: local subprocess contract


def test_local_ok():
    res = LOCAL.run({"hello.py": "print('hi')"}, ["python", "hello.py"], timeout=5.0)
    assert res.status is sbx.RunStatus.OK
    assert res.exit_code == 0
    assert res.stdout.strip() == "hi"
    assert res.stderr == ""
    assert not res.timed_out
    assert not res.oom_killed


def test_local_failed_exit_one():
    res = LOCAL.run({"boom.py": "raise SystemExit(1)"}, ["python", "boom.py"], timeout=5.0)
    assert res.status is sbx.RunStatus.FAILED
    assert res.exit_code == 1


def test_local_crashed_signal():
    res = LOCAL.run({"die.py": "import os; os.abort()"}, ["python", "die.py"], timeout=5.0)
    assert res.status is sbx.RunStatus.CRASHED
    assert res.exit_code == -signal.SIGABRT


def test_local_timed_out():
    res = LOCAL.run({"slow.py": "import time; time.sleep(30)"}, ["python", "slow.py"], timeout=1.5)
    assert res.status is sbx.RunStatus.TIMED_OUT
    assert res.exit_code is None
    assert res.timed_out
    assert res.duration < 30  # the group kill bounds the wait


def test_local_stages_nested_files():
    res = LOCAL.run(
        {"pkg/mod.py": "X = 1", "main.py": "from pkg.mod import X; print(X)"},
        ["python", "main.py"],
        timeout=5.0,
    )
    assert res.status is sbx.RunStatus.OK
    assert res.stdout.strip() == "1"


def test_local_sets_deterministic_env(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "random")  # the executor must override
    src = "import os; print(os.environ['PYTHONHASHSEED'])"
    res = LOCAL.run({"env.py": src}, ["python", "env.py"], timeout=5.0)
    assert res.stdout.strip() == "0"


# =========================================== tier b: Docker-gated smoke tests


def _docker_image_present() -> bool:
    rc, _out, _err = sbx._docker(["image", "inspect", sbx.IMAGE])
    return rc == 0


_docker_mark = pytest.mark.skipif(not _docker_image_present(), reason="sandbox image not built")


@_docker_mark
def test_docker_smoke_ok():
    res = sbx.DockerSandbox().run({"hello.py": "print('hi')"}, ["python", "hello.py"], timeout=10.0)
    assert res.status is sbx.RunStatus.OK
    assert res.stdout.strip() == "hi"


@_docker_mark
def test_docker_smoke_timeout():
    res = sbx.DockerSandbox().run(
        {"slow.py": "import time; time.sleep(30)"}, ["python", "slow.py"], timeout=1.5
    )
    assert res.status is sbx.RunStatus.TIMED_OUT


@_docker_mark
def test_docker_smoke_read_only_root():
    # The root filesystem is read-only: a write to / must fail.
    res = sbx.DockerSandbox().run(
        {"w.py": "open('/evil.txt', 'w')"}, ["python", "w.py"], timeout=10.0
    )
    assert res.status is not sbx.RunStatus.OK
