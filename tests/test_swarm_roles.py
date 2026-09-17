"""Role-client tests: parsing, prompt isolation, and the Anthropic adapter.

No API in CI: the adapter tests run against a stub client object. The one
live round-trip is skipped unless ``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from evogine.engine.records import LearningRecord
from evogine.swarm.roles import (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_MAX_RETRIES,
    DEFAULT_READ_TIMEOUT_S,
    AnthropicModel,
    ModelConfig,
    ModelResponse,
    build_builder_messages,
    build_reflector_messages,
    build_verifier_messages,
    extract_code,
    parse_learning_draft,
    parse_verdict,
)


class StubMessages:
    """Records create() kwargs; returns one thinking block and one text block."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="step-by-step reasoning"),
                SimpleNamespace(type="text", text="REPAIRED SOURCE"),
            ],
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            model="stub-model",
            stop_reason="end_turn",
        )


class StubClient:
    def __init__(self) -> None:
        self.messages = StubMessages()


def test_extract_code_fenced_uses_last_block() -> None:
    text = (
        "Here is my repair:\n```python\nfirst()\n```\n"
        "And the final source:\n```python\nfinal()\n```"
    )
    assert extract_code(text) == "final()"


def test_extract_code_plain_reply_is_the_source() -> None:
    assert extract_code("def fetch():\n    return 1\n") == "def fetch():\n    return 1"


def test_extract_code_json_fence_keeps_payload_tag_free() -> None:
    """A package reply fences its {"files": ...} payload as ```json; the tag
    must be consumed by the fence, not ride into the extracted content (live
    pilot: the python-only tag alternation leaked ``json\n`` into the payload
    and every package-layout candidate failed extraction)."""
    text = '```json\n{"files": {"a.py": "x = 1\\n"}}\n```'
    assert extract_code(text) == '{"files": {"a.py": "x = 1\\n"}}'


def test_parse_verdict_pass_fail_unknown() -> None:
    assert parse_verdict("VERDICT: pass\nfine under pressure") == "pass"
    assert parse_verdict("verdict: fail\nraises OverflowError") == "fail"
    assert parse_verdict("looks good to me") == "unknown"


def test_parse_learning_draft_plain_fenced_and_invalid() -> None:
    assert parse_learning_draft('{"type": "failure-pattern", "title": "t"}') == {
        "type": "failure-pattern",
        "title": "t",
    }
    fenced = '```json\n{"title": "t", "confidence": 0.4}\n```'
    assert parse_learning_draft(fenced)["title"] == "t"
    with pytest.raises(ValueError):
        parse_learning_draft("no json at all")


def test_builder_prompt_carries_knowledge_but_never_failure_details() -> None:
    record = LearningRecord.sample()
    task = SimpleNamespace(
        case_id="case_retries",
        decay_axis="retries",
        description="Retry handling under 429 pressure",
        scenario_brief="Sudden 429 bursts",
        layout="single",
        baseline_files=MappingProxyType({"candidate.py": "def fetch():\n    return 1\n"}),
    )
    messages = build_builder_messages(task, [record])
    system, user = messages[0]["content"], messages[1]["content"]
    assert "Promoted knowledge to apply:" in user
    assert "Unhandled 429 backoff" in user  # the record's rendered knowledge
    assert "repair builder" in system
    # The standing leak guard: no gate failure details in builder prompts.
    assert "hidden property" not in user
    assert "expected" not in user.lower()


def test_verifier_prompt_sees_candidate_not_knowledge() -> None:
    task = SimpleNamespace(
        case_id="case_retries",
        decay_axis="retries",
        description="Retry handling",
        scenario_brief="429 bursts",
        layout="single",
        baseline_files={"candidate.py": "def fetch():\n    return 1\n"},
    )
    messages = build_verifier_messages(task, "def fetch():\n    return 2\n")
    assert "independent verifier" in messages[0]["content"]
    assert "Proposed repair to review:" in messages[1]["content"]
    assert "def fetch():\n    return 2\n" in messages[1]["content"]


def test_reflector_prompt_is_the_diagnosis_surface() -> None:
    task = SimpleNamespace(
        case_id="case_retries",
        decay_axis="retries",
        description="Retry handling",
        scenario_brief="429 bursts",
        layout="single",
        baseline_files={"candidate.py": "def fetch():\n    return 1\n"},
    )
    evidence = "candidate gen=0 idx=0: failure=hidden property p_x expected 3"
    messages = build_reflector_messages(task, evidence)
    # The reflector is the diagnosis role: failure details travel here by design.
    assert "hidden property p_x" in messages[1]["content"]
    assert "single JSON object" in messages[0]["content"]


def test_anthropic_model_splits_system_and_separates_reasoning() -> None:
    client = StubClient()
    adapter = AnthropicModel(client=client)
    response = adapter.generate(
        messages=[
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "USER PROMPT"},
        ],
        model_config=ModelConfig(
            model="stub-model", temperature=0.0, thinking=False, seed="ab12cd34", max_tokens=99
        ),
    )
    kwargs = client.messages.kwargs
    assert kwargs is not None
    assert kwargs["system"] == "SYSTEM PROMPT"
    assert kwargs["messages"] == [{"role": "user", "content": "USER PROMPT"}]
    # anthropic>=1.6 removed the typed temperature kwarg; the playbook pins
    # temperature through extra_body instead.
    assert "temperature" not in kwargs
    assert kwargs["extra_body"] == {"temperature": 0.0}
    assert kwargs["max_tokens"] == 99
    assert kwargs["thinking"] == {"type": "disabled"}
    assert response.text == "REPAIRED SOURCE"
    assert response.reasoning_content == "step-by-step reasoning"
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert response.latency_ms >= 0.0
    assert response.stop_reason == "end_turn"


def test_anthropic_model_thinking_on_pins_temperature() -> None:
    client = StubClient()
    AnthropicModel(client=client).generate(
        messages=[{"role": "user", "content": "go"}],
        model_config=ModelConfig(
            model="stub-model", temperature=0.0, thinking=True, seed="ab12cd34"
        ),
    )
    kwargs = client.messages.kwargs
    assert kwargs is not None
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    # The API's constraint; learn_0012 measured thinking-on as nondeterministic.
    # Regression: anthropic 1.6.0 raises TypeError on a typed temperature kwarg
    # (crashed every live run before any HTTP request); temperature must ride
    # through extra_body instead.
    assert "temperature" not in kwargs
    assert kwargs["extra_body"] == {"temperature": 1.0}


def test_model_response_defaults() -> None:
    response = ModelResponse(text="x")
    assert response.reasoning_content == ""
    assert response.input_tokens is None
    assert response.output_tokens is None


class _SilentServer:
    """A TCP server that accepts connections and never answers.

    Stand-in for the observed phase-2 fault: an established connection whose
    edge stops sending, so the request hangs in the read phase with no error
    surfacing. The socket is real (not a mock transport) because the read
    timeout is enforced by the transport layer, exactly as in a live hang.
    """

    def __enter__(self) -> str:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        host, port = self._sock.getsockname()
        return f"http://{host}:{port}"

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
                conn.recv(4096)  # drain the request, never reply
            except OSError:
                return  # server socket closed

    def __exit__(self, *args: object) -> None:
        self._sock.close()


def test_hung_request_surfaces_the_typed_timeout() -> None:
    """A request hung on an established connection raises the SDK's typed
    timeout within the configured read bound — the fault the phase-2 benchmark
    stalled 20+ minutes without when no read bound was effectively in force
    (the default retry ladder drained silently)."""
    with _SilentServer() as base_url:
        # max_retries=0 keeps the bound exactly one read timeout; the default
        # retry budget is pinned separately below.
        client = anthropic.Anthropic(
            api_key="test-key",
            base_url=base_url,
            timeout=httpx2.Timeout(connect=2.0, read=0.5, write=2.0, pool=2.0),
            max_retries=0,
        )
        model = AnthropicModel(client=client)
        started = time.perf_counter()
        with pytest.raises(anthropic.APITimeoutError):
            model.generate(
                messages=[{"role": "user", "content": "go"}],
                model_config=ModelConfig(
                    model="claude-sonnet-4-5", temperature=0.0, thinking=False, seed="ab12cd34"
                ),
            )
    # Bounded, not the observed silent hang; generous slop for slow CI hosts.
    assert time.perf_counter() - started < 15


def test_default_client_wires_the_configured_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructor knobs reach the SDK client: custom read timeout honored,
    safe defaults explicit — never the SDK's implicit bare defaults."""
    captured: dict[str, Any] = {}

    class RecordingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", RecordingClient)

    # Defaults: short connect, the documented 600s generation ceiling, one
    # bounded retry.
    AnthropicModel()._default_client()
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx2.Timeout)
    assert timeout.connect == DEFAULT_CONNECT_TIMEOUT_S == 10.0
    assert timeout.read == DEFAULT_READ_TIMEOUT_S == 600.0
    assert captured["max_retries"] == DEFAULT_MAX_RETRIES == 1

    # The knob is honored: a custom read timeout reaches the SDK client.
    captured.clear()
    AnthropicModel(read_timeout=321.0, connect_timeout=4.5, max_retries=2)._default_client()
    custom = captured["timeout"]
    assert isinstance(custom, httpx2.Timeout)
    assert custom.connect == 4.5
    assert custom.read == 321.0
    assert captured["max_retries"] == 2


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires ANTHROPIC_API_KEY — the live-API integration path",
)
def test_live_api_roundtrip() -> None:
    response = AnthropicModel().generate(
        messages=[
            {"role": "system", "content": "Reply with the single word OK and nothing else."},
            {"role": "user", "content": "hello"},
        ],
        model_config=ModelConfig(
            model="claude-sonnet-4-5",
            temperature=0.0,
            thinking=False,
            seed="beef0001",
            max_tokens=16,
        ),
    )
    assert response.text.strip()
    assert response.output_tokens is not None
