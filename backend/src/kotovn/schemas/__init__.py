"""Data contracts shared across engine modules and the desktop client."""

from __future__ import annotations

from kotovn.schemas.dialogue import (
    BoundingBox,
    DialogueFrame,
    GlossaryHit,
    PipelineTelemetry,
    SourceType,
    TranslationResult,
    VisualContext,
)
from kotovn.schemas.persona import (
    ArchetypeTemplate,
    LinguisticFingerprint,
    PersonaCard,
    PersonaOrigin,
    StyleAnchor,
)
from kotovn.schemas.rag import (
    EntryStatus,
    EpisodeChunk,
    GlossaryEntry,
    KnowledgeSnapshot,
    RAGContext,
    ReflectionProposal,
    RelationEdge,
)

__all__ = [
    "ArchetypeTemplate",
    "BoundingBox",
    "DialogueFrame",
    "EntryStatus",
    "EpisodeChunk",
    "GlossaryEntry",
    "GlossaryHit",
    "KnowledgeSnapshot",
    "LinguisticFingerprint",
    "PersonaCard",
    "PersonaOrigin",
    "PipelineTelemetry",
    "RAGContext",
    "ReflectionProposal",
    "RelationEdge",
    "SourceType",
    "StyleAnchor",
    "TranslationResult",
    "VisualContext",
]
