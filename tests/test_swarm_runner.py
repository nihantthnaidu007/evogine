"""Swarm loop tests: a FakeModel drives the full loop — no API in CI.

Covers the blueprint lifecycle end to end against the real git-backed
Engine: warm retrieval of seeded knowledge, best-of-N retention with the
never-worse fallback, the failure path with the reflector's proposal landing
ephemeral via ``Engine.propose``, and phase0 rollout-record completeness.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from evogine.engine.bootstrap import load_seeds
from evogine.engine.store import Engine
from evogine.swarm.roles import ModelConfig, ModelResponse
from evogine.swarm.rollouts import ROLLOUT_FIELDS
from evogine.swarm.runner import GateOutcome, SwarmConfig, SwarmTask, derive_seed, run_task

SEEDS_DIR = Path(__file__).resolve().parents[1] / "src" / "evogine" / "seeds"

BASELINE_SOURCE = "def fetch(url, retries=1):\n    return 'ok'\n"
FIXED_SOURCE = (
    "def fetch(url, retries=1):\n    # FIXED: bounded jittered backoff\n    return 'ok'\n"
)
BROKEN_SOURCE = "def fetch(url, retries=999):\n    raise OverflowError('unbounded')\n"

REFLECTOR_JSON = json.dumps(
    {
        "type": "failure-pattern",
        "title": "Unbounded retry loops amplify 429 pressure",
        "confidence": 0.5,
        "body": {
            "signal": "Candidates crashed on overflow",
            "failure_mode": "Retries without a budget amplify pressure",
            "detection": "Gate failure on the hidden overflow property",
            "fix_strategy": "Bound retries with jittered backoff",
        },
    }
)


def fixed_reply(messages: list[dict[str, str]]) -> str:
    return FIXED_SOURCE


def broken_reply(messages: list[dict[str, str]]) -> str:
    return BROKEN_SOURCE


class FakeModel:
    """Scripted ModelClient routed on the per-role model name; thread-safe.

    The builder reply is a callable so each test scripts candidate quality;
    the verifier always passes; the reflector replies with record JSON.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self.builder_messages: list[list[dict[str, str]]] = []
        self.active = 0
        self.max_observed_parallel = 0
        self.builder_reply: Callable[[list[dict[str, str]]], str] = fixed_reply
        self.verifier_reply = "VERDICT: pass\nHolds the contract under pressure."
        self.reflector_reply = REFLECTOR_JSON

    def generate(
        self, *, messages: list[dict[str, str]], model_config: ModelConfig
    ) -> ModelResponse:
        with self._lock:
            self.calls.append((model_config.model, messages))
            self.active += 1
            self.max_observed_parallel = max(self.max_observed_parallel, self.active)
        try:
            name = model_config.model
            if name.endswith("builder"):
                self.builder_messages.append(messages)
                text = self.builder_reply(messages)
            elif name.endswith("verifier"):
                text = self.verifier_reply
            else:
                text = self.reflector_reply
            return ModelResponse(
                text=text,
                reasoning_content="",
                input_tokens=10,
                output_tokens=5,
                latency_ms=1.5,
                model=model_config.model,
                stop_reason="end_turn",
            )
        finally:
            with self._lock:
                self.active -= 1


@dataclass
class FakeGate:
    """Scripted CandidateGate.

    The degraded baseline passes the visible suite and 1/3 hidden properties.
    Candidates carrying the FIX marker pass 3/3 (or 0/3 when
    ``worse_candidates`` pins them under the baseline); anything else fails a
    visible test with an oracle-naming failure detail.
    """

    fail_candidates: bool = False
    worse_candidates: bool = False
    calls: list[Mapping[str, str]] = field(default_factory=list)

    def evaluate(self, files: Mapping[str, str], task: SwarmTask) -> GateOutcome:
        self.calls.append(dict(files))
        if files == task.baseline_files:
            return GateOutcome(
                verdict="pass",
                failure_detail=None,
                visible_passed=2,
                visible_total=2,
                properties_passed=1,
                properties_total=3,
                ops={
                    "visible_tests": 2,
                    "visible_failures": 0,
                    "properties_checked": 3,
                    "properties_failed": 2,
                },
            )
        source = files.get("candidate.py", "")
        if self.fail_candidates or "FIXED" not in source:
            return GateOutcome(
                verdict="fail",
                failure_detail="hidden property p_overflow expected no raise; got OverflowError",
                visible_passed=1,
                visible_total=2,
                properties_passed=0,
                properties_total=3,
                ops={
                    "visible_tests": 2,
                    "visible_failures": 1,
                    "properties_checked": 3,
                    "properties_failed": 3,
                },
            )
        passed = 0 if self.worse_candidates else 3
        return GateOutcome(
            verdict="pass",
            failure_detail=None,
            visible_passed=2,
            visible_total=2,
            properties_passed=passed,
            properties_total=3,
            ops={
                "visible_tests": 2,
                "visible_failures": 0,
                "properties_checked": 3,
                "properties_failed": 3 - passed,
            },
        )


def make_task() -> SwarmTask:
    return SwarmTask(
        case_id="case_retries",
        description="Retry handling under 429 pressure",
        scenario_brief="Sudden 429 bursts overwhelm the single retry",
        baseline_files={"candidate.py": BASELINE_SOURCE},
        decay_axis="retries",
    )


def make_config(tmp_path: Path, **overrides: object) -> SwarmConfig:
    settings: dict[str, object] = {
        "run_id": "runswarm01",
        "builder_model": "fake-builder",
        "verifier_model": "fake-verifier",
        "reflector_model": "fake-reflector",
        "candidates_per_generation": 2,
        "runs_dir": tmp_path / "runs",
    }
    settings.update(overrides)
    return SwarmConfig(**settings)  # type: ignore[arg-type]


def seeded_engine(tmp_path: Path) -> Engine:
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    return engine


def read_rollouts(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_seed_derivation_matches_learn_0012() -> None:
    expected = hashlib.sha256(b"runswarm01|case_retries|0|1").hexdigest()[:8]
    assert derive_seed("runswarm01", "case_retries", 0, 1) == expected


def test_warm_run_retrieves_seeded_knowledge_and_retains_candidate(tmp_path: Path) -> None:
    engine = seeded_engine(tmp_path)
    fake = FakeModel()
    report = run_task(make_task(), engine, fake, FakeGate(), make_config(tmp_path))

    # Warm retrieval: the "case"/"retry" tokens match the seed pack; top-k fill follows.
    assert report["retrieved_records"] == 5
    assert "learn_0001" in report["retrieved_ids"]
    # Knowledge rendered into the builder prompt (warm arm) — the top-ranked
    # seed's title must be visible to the builder.
    builder_user = fake.builder_messages[0][1]["content"]
    assert "Promoted knowledge to apply:" in builder_user
    assert "No improvement claim below the per-case noise floor" in builder_user
    # The verifier's independent voice never sees the knowledge block.
    verifier_messages = [messages for name, messages in fake.calls if name.endswith("verifier")]
    assert verifier_messages
    assert "Promoted knowledge" not in verifier_messages[0][1]["content"]
    # Best-of-N retention.
    assert report["retained"] == "candidate"
    assert report["best"] is not None
    assert report["best"]["properties_passed"] == 3
    assert report["proposal"] is None
    # No failure, no reflector call.
    stages = [record["stage"] for record in read_rollouts(tmp_path / "runs" / "rollouts.jsonl")]
    assert "reflector" not in stages


def test_failure_path_reflector_proposal_lands_ephemeral(tmp_path: Path) -> None:
    engine = seeded_engine(tmp_path)
    fake = FakeModel()
    fake.builder_reply = broken_reply
    report = run_task(make_task(), engine, fake, FakeGate(), make_config(tmp_path))

    assert report["retained"] == "baseline"
    assert report["best"] is None
    assert report["proposal"] is not None
    learning_id = report["proposal"]["learning_id"]
    assert isinstance(learning_id, str)
    record = engine.get(learning_id)
    # The proposal landed through the engine's lifecycle: ephemeral, never promoted.
    assert record.status == "ephemeral"
    assert record.type == "failure-pattern"
    assert record.provenance.proposed_by == "reflector"
    assert record.provenance.origin == "swarm-runner"
    # Promoted-only retrieval: the fresh record is invisible to every query.
    assert all(seen.id != record.id for seen in engine.retrieve(record.title, k=50))
    # The reflector step is on the audit trail with the learning id.
    rollouts = read_rollouts(tmp_path / "runs" / "rollouts.jsonl")
    reflector_records = [r for r in rollouts if r["stage"] == "reflector"]
    assert len(reflector_records) == 1
    assert reflector_records[0]["verdict"] == "proposed"
    assert reflector_records[0]["config"]["learning_id"] == record.id
    assert reflector_records[0]["extracted_code"] == REFLECTOR_JSON


def test_best_of_n_keeps_baseline_when_all_candidates_worse(tmp_path: Path) -> None:
    engine = seeded_engine(tmp_path)
    fake = FakeModel()  # builders return FIXED source: visible pass, but...
    report = run_task(
        make_task(),
        engine,
        fake,
        FakeGate(worse_candidates=True),  # ...0/3 properties vs the baseline's 1/3
        make_config(tmp_path),
    )
    assert report["retained"] == "baseline"
    assert report["best"] is None
    # The candidates were not broken — they were strictly worse on properties.
    assert all(candidate["verdict"] == "pass" for candidate in report["candidates"])
    assert all(candidate["properties_passed"] == 0 for candidate in report["candidates"])


def test_rollout_records_carry_full_phase0_schema(tmp_path: Path) -> None:
    engine = seeded_engine(tmp_path)
    fake = FakeModel()
    fake.builder_reply = broken_reply  # drive every stage: baseline+candidates+reflector
    report = run_task(make_task(), engine, fake, FakeGate(), make_config(tmp_path, generations=2))
    rollouts = read_rollouts(tmp_path / "runs" / "rollouts.jsonl")
    assert len(rollouts) == 1 + 2 * 2 * 2 + 1  # baseline + (candidate+verifier)*4 + reflector

    for record in rollouts:
        assert set(record) == set(ROLLOUT_FIELDS), f"incomplete record: {sorted(record)}"
        assert record["timestamp"]
        assert record["case_id"] == "case_retries"
        assert record["run_id"] == "runswarm01"

    baseline = rollouts[0]
    assert baseline["stage"] == "baseline"
    assert baseline["generation"] is None and baseline["candidate_index"] is None
    assert baseline["config"]["seed"] is None
    assert baseline["config"]["playbook"] == "learn_0012"
    assert baseline["prompt_messages"] == []

    candidates = [r for r in rollouts if r["stage"] == "candidate"]
    verifiers = [r for r in rollouts if r["stage"] == "verifier"]
    for record in candidates:
        seed = derive_seed(
            "runswarm01", "case_retries", record["generation"], record["candidate_index"]
        )
        assert record["config"]["seed"] == seed
        assert record["config"]["playbook"] == "learn_0012"
        assert record["config"]["temperature"] == 0.0
        assert record["config"]["thinking"] is False
        assert record["model"] == "fake-builder"
        assert record["prompt_messages"][0]["role"] == "system"
        assert record["tokens"] == {"input_tokens": 10, "output_tokens": 5}
        assert record["latency_ms"] >= 0.0
        assert record["reasoning_content"] == ""
        assert record["ops"]["visible_tests"] == 2
    for record in verifiers:
        assert record["model"] == "fake-verifier"
        seed = derive_seed(
            "runswarm01", "case_retries", record["generation"], record["candidate_index"]
        )
        assert record["config"]["seed"] == seed  # seed-paired with its candidate
        assert record["verdict"] == "pass"
        assert record["failure_detail"] is None

    reflector_record = rollouts[-1]
    assert reflector_record["stage"] == "reflector"
    assert reflector_record["generation"] is None
    # The broken reply drove every candidate out, so the reflector proposed.
    proposal = report["proposal"]
    assert proposal is not None
    assert reflector_record["config"]["learning_id"] == proposal["learning_id"]
    assert reflector_record["latency_ms"] >= 0.0

    # Rollout records append in deterministic generation/index order.
    order = [(r["generation"], r["candidate_index"]) for r in candidates]
    assert order == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_failure_details_never_reach_builder_prompts(tmp_path: Path) -> None:
    engine = seeded_engine(tmp_path)
    fake = FakeModel()
    fake.builder_reply = broken_reply
    run_task(make_task(), engine, fake, FakeGate(), make_config(tmp_path))
    rollouts = read_rollouts(tmp_path / "runs" / "rollouts.jsonl")

    candidate_prompt_text = " ".join(
        message["content"]
        for record in rollouts
        if record["stage"] == "candidate"
        for message in record["prompt_messages"]
    )
    # The gate names a hidden property; no builder prompt may ever see it.
    assert "hidden property" not in candidate_prompt_text
    assert "p_overflow" not in candidate_prompt_text
    reflector_record = next(r for r in rollouts if r["stage"] == "reflector")
    assert "p_overflow" in " ".join(m["content"] for m in reflector_record["prompt_messages"])


def test_max_parallel_builders_bounds_concurrency(tmp_path: Path) -> None:
    engine = seeded_engine(tmp_path)

    # Two builders must actually overlap: the barrier only releases in pairs.
    barrier = threading.Barrier(2, timeout=10)

    def paired_reply(messages: list[dict[str, str]]) -> str:
        barrier.wait()
        return FIXED_SOURCE

    fake_paired = FakeModel()
    fake_paired.builder_reply = paired_reply
    run_task(
        make_task(),
        engine,
        fake_paired,
        FakeGate(),
        make_config(tmp_path, max_parallel_builders=2, candidates_per_generation=4),
    )
    assert fake_paired.max_observed_parallel == 2

    # The serial default never overlaps.
    fake_serial = FakeModel()
    run_task(
        make_task(),
        engine,
        fake_serial,
        FakeGate(),
        make_config(tmp_path, max_parallel_builders=1, candidates_per_generation=4),
    )
    assert fake_serial.max_observed_parallel == 1


def test_unparseable_reflector_reply_is_recorded_not_landed(tmp_path: Path) -> None:
    engine = seeded_engine(tmp_path)
    fake = FakeModel()
    fake.builder_reply = broken_reply
    fake.reflector_reply = "I could not produce JSON, sorry."
    report = run_task(make_task(), engine, fake, FakeGate(), make_config(tmp_path))

    assert report["retained"] == "baseline"
    assert report["proposal"] is None
    rollouts = read_rollouts(tmp_path / "runs" / "rollouts.jsonl")
    reflector_record = next(r for r in rollouts if r["stage"] == "reflector")
    assert reflector_record["verdict"] == "fail"
    assert "learning proposal rejected" in reflector_record["failure_detail"]
    # Nothing beyond the 12 seeds landed in the store.
    assert len(list(engine.records_dir.glob("*.json"))) == 12
