"""Role-client tests: parsing, prompt isolation, and the Anthropic adapter.

No API in CI: the adapter tests run against a stub client object. The one
live round-trip is skipped unless ``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

import os
from types import MappingProxyType, SimpleNamespace

import pytest

from evogine.engine.records import LearningRecord
from evogine.swarm.roles import (
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
    assert kwargs["temperature"] == 0.0
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
    assert kwargs["temperature"] == 1.0


def test_model_response_defaults() -> None:
    response = ModelResponse(text="x")
    assert response.reasoning_content == ""
    assert response.input_tokens is None
    assert response.output_tokens is None


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
