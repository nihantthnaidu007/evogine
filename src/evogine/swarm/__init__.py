"""Swarm: the task-scaled agent roles the engine coordinates."""

from __future__ import annotations

from evogine.swarm.gates_adapter import CodeGateAdapter
from evogine.swarm.roles import (
    AnthropicModel,
    Builder,
    ModelClient,
    ModelConfig,
    ModelResponse,
    Reflector,
    RoleCall,
    Verifier,
    build_builder_messages,
    build_reflector_messages,
    build_verifier_messages,
    extract_code,
    parse_learning_draft,
    parse_verdict,
    render_knowledge,
)
from evogine.swarm.rollouts import ROLLOUT_FIELDS, RolloutWriter, build_rollout
from evogine.swarm.runner import (
    PLAYBOOK,
    CandidateGate,
    EngineLike,
    GateOutcome,
    SwarmConfig,
    SwarmTask,
    derive_seed,
    run_task,
)

__all__ = [
    "PLAYBOOK",
    "ROLLOUT_FIELDS",
    "AnthropicModel",
    "Builder",
    "CandidateGate",
    "CodeGateAdapter",
    "EngineLike",
    "GateOutcome",
    "ModelClient",
    "ModelConfig",
    "ModelResponse",
    "Reflector",
    "RoleCall",
    "RolloutWriter",
    "SwarmConfig",
    "SwarmTask",
    "Verifier",
    "build_builder_messages",
    "build_reflector_messages",
    "build_rollout",
    "build_verifier_messages",
    "derive_seed",
    "extract_code",
    "parse_learning_draft",
    "parse_verdict",
    "render_knowledge",
    "run_task",
]
