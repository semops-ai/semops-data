"""
Episode model for Episode-Centric Provenance.

An Episode represents one meaningful operation that modifies the DDD layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import hashlib
import ulid


class OperationType(str, Enum):
    """Types of operations that create episodes."""

    # Content operations
    INGEST = "ingest"  # entity created from source document
    CLASSIFY = "classify"  # entity assigned primary pattern
    DECLARE_PATTERN = "declare_pattern"  # new pattern created from synthesis
    PUBLISH = "publish"  # delivery created
    SYNTHESIZE = "synthesize"  # new content generated from multiple sources
    CREATE_EDGE = "create_edge"  # relationship established
    EMBED = "embed"  # embedding generated
    # Governance operations
    AUDIT = "audit"  # checked state against expected state
    EVALUATE = "evaluate"  # assessed input against domain model
    MEASURE = "measure"  # computed coherence/drift score
    SYNC = "sync"  # updated state to match authority
    BRIDGE = "bridge"  # mapped concepts to patterns (HITL)


class TargetType(str, Enum):
    """Types of targets that can be modified."""

    ENTITY = "entity"
    PATTERN = "pattern"
    EDGE = "edge"
    DELIVERY = "delivery"
    CORPUS = "corpus"
    REPO = "repo"


class ReasoningPattern(str, Enum):
    """Agentic reasoning strategies (vocabulary from semops-orchestrator#202)."""

    WORKFLOW = "workflow"  # sequential steps, no branching
    COT = "cot"  # chain of thought: sequential reasoning, single path
    REACT = "react"  # observation-action cycles with tool use
    TREE_OF_THOUGHTS = "tree-of-thoughts"  # branching exploration with backtracking
    REFLEXION = "reflexion"  # self-critique and revision loops
    LLM_P = "llm-p"  # LLM-based planning
    DIRECT = "direct"  # no explicit reasoning strategy (single-shot)


class ContextAssemblyMethod(str, Enum):
    """How context was constructed for an operation."""

    RAG = "rag"  # retrieval-augmented generation
    FULL_DOC = "full_doc"  # full document(s) loaded
    SUMMARY = "summary"  # pre-summarized context
    HYBRID = "hybrid"  # combination of methods


@dataclass
class DetectedEdge:
    """A relationship detected/proposed by an agent."""

    predicate: str  # derived_from, cites, related_to, etc.
    target_id: str  # ID of the related artifact
    strength: float = 1.0  # 0.0-1.0
    rationale: str | None = None  # Why the agent thinks this relationship exists

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "target_id": self.target_id,
            "strength": self.strength,
            "rationale": self.rationale,
        }


@dataclass
class Episode:
    """
    An episode in the provenance chain.

    Episodes are created automatically when DDD-touching operations execute.
    They capture the full context needed for "why was this classified this way?" audits.
    """

    # Required fields
    operation: OperationType
    target_type: TargetType
    target_id: str

    # Auto-generated
    id: str = field(default_factory=lambda: str(ulid.new()))
    created_at: datetime = field(default_factory=datetime.now)

    # Run context (set by LineageTracker)
    run_id: str | None = None

    # Context used (for classification/declaration audits)
    context_pattern_ids: list[str] = field(default_factory=list)
    context_entity_ids: list[str] = field(default_factory=list)

    # Quality signals
    coherence_score: float | None = None

    # Reasoning trace (ADR-0017, #160)
    reasoning_pattern: ReasoningPattern | None = None
    chain_depth: int | None = None
    branching_factor: int | None = None
    observation_action_cycles: int | None = None
    context_assembly_method: ContextAssemblyMethod | None = None
    context_token_count: int | None = None
    context_utilization: float | None = None  # 0.0-1.0

    # Agent info
    agent_name: str | None = None
    agent_version: str | None = None
    model_name: str | None = None
    prompt_hash: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)

    # Detected edges
    detected_edges: list[DetectedEdge] = field(default_factory=list)

    # Metadata
    input_hash: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_input_hash(self, content: str) -> str:
        """Compute SHA256 hash of input for deduplication."""
        self.input_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
        return self.input_hash

    def add_context_pattern(self, pattern_id: str) -> None:
        """Add a pattern to the context (retrieved/considered during operation)."""
        if pattern_id not in self.context_pattern_ids:
            self.context_pattern_ids.append(pattern_id)

    def add_context_entity(self, entity_id: str) -> None:
        """Add an entity to the context (used during operation)."""
        if entity_id not in self.context_entity_ids:
            self.context_entity_ids.append(entity_id)

    def add_detected_edge(
        self,
        predicate: str,
        target_id: str,
        strength: float = 1.0,
        rationale: str | None = None,
    ) -> None:
        """Add a model-detected relationship."""
        self.detected_edges.append(
            DetectedEdge(
                predicate=predicate,
                target_id=target_id,
                strength=strength,
                rationale=rationale,
            )
        )

    def set_reasoning_trace(
        self,
        pattern: ReasoningPattern | str,
        chain_depth: int | None = None,
        branching_factor: int | None = None,
        observation_action_cycles: int | None = None,
        context_assembly_method: ContextAssemblyMethod | str | None = None,
        context_token_count: int | None = None,
        context_utilization: float | None = None,
    ) -> None:
        """Set reasoning strategy metadata for this episode."""
        self.reasoning_pattern = (
            ReasoningPattern(pattern) if isinstance(pattern, str) else pattern
        )
        self.chain_depth = chain_depth
        self.branching_factor = branching_factor
        self.observation_action_cycles = observation_action_cycles
        if context_assembly_method is not None:
            self.context_assembly_method = (
                ContextAssemblyMethod(context_assembly_method)
                if isinstance(context_assembly_method, str)
                else context_assembly_method
            )
        self.context_token_count = context_token_count
        self.context_utilization = context_utilization

    def set_agent_info(
        self,
        name: str,
        version: str | None = None,
        model: str | None = None,
        prompt_hash: str | None = None,
    ) -> None:
        """Set agent metadata for reproducibility."""
        self.agent_name = name
        self.agent_version = version
        self.model_name = model
        self.prompt_hash = prompt_hash

    def set_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        """Record token usage for cost tracking."""
        self.token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "operation": self.operation.value,
            "target_type": self.target_type.value,
            "target_id": self.target_id,
            "context_pattern_ids": self.context_pattern_ids,
            "context_entity_ids": self.context_entity_ids,
            "coherence_score": self.coherence_score,
            "reasoning_pattern": self.reasoning_pattern.value if self.reasoning_pattern else None,
            "chain_depth": self.chain_depth,
            "branching_factor": self.branching_factor,
            "observation_action_cycles": self.observation_action_cycles,
            "context_assembly_method": self.context_assembly_method.value if self.context_assembly_method else None,
            "context_token_count": self.context_token_count,
            "context_utilization": self.context_utilization,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "model_name": self.model_name,
            "prompt_hash": self.prompt_hash,
            "token_usage": self.token_usage,
            "detected_edges": [e.to_dict() for e in self.detected_edges],
            "input_hash": self.input_hash,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
