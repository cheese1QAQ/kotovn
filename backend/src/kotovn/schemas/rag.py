"""Schemas for Hybrid RAG knowledge storage and online maintenance ('learn while translating')."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class EntryStatus(str, Enum):
    """Lifecycle status of an automatically learned or user-verified knowledge entry."""

    DRAFT = "draft"
    VERIFIED = "verified"


class GlossaryEntry(BaseModel):
    """Exact-match term entry (names, places, world-building terms, honorifics)."""

    source_term: str = Field(min_length=1)
    target_term: str = Field(min_length=1)
    category: str = Field(default="term", description="character | location | skill | honorific")
    status: EntryStatus = Field(default=EntryStatus.DRAFT)
    note: str = Field(default="")
    evidence_frame_id: str | None = Field(default=None)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RelationEdge(BaseModel):
    """Directed interpersonal relation and addressing convention between two characters."""

    source_character: str = Field(min_length=1)
    target_character: str = Field(min_length=1)
    addressing_term: str = Field(
        description="How source addresses target, e.g., '前辈', '笨蛋老哥'"
    )
    relation_label: str = Field(default="acquaintance")
    politeness: str = Field(default="casual", description="formal | polite | casual | rough")
    status: EntryStatus = Field(default=EntryStatus.DRAFT)


class EpisodeChunk(BaseModel):
    """Scene or dialogue window memory stored for semantic/lexical retrieval."""

    chunk_id: str
    summary: str = Field(min_length=1)
    speakers: list[str] = Field(default_factory=list)
    scene_tag: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RAGContext(BaseModel):
    """Aggregated hybrid retrieval payload injected into the translation prompt."""

    glossary: list[GlossaryEntry] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    episodes: list[EpisodeChunk] = Field(default_factory=list)


class ReflectionProposal(BaseModel):
    """Structured delta extracted asynchronously by the background Reflector."""

    glossary_candidates: list[GlossaryEntry] = Field(default_factory=list)
    relation_updates: list[RelationEdge] = Field(default_factory=list)
    episode_summary: EpisodeChunk | None = Field(default=None)


class KnowledgeSnapshot(BaseModel):
    """Checkpoint enabling safe Save/Load (SL) and route-branch rollback."""

    snapshot_id: str
    label: str
    route_name: str = Field(default="common")
    glossary_count: int = Field(ge=0)
    relation_count: int = Field(ge=0)
    episode_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
