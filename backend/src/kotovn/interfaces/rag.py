"""Protocols for Hybrid RAG storage and asynchronous online reflection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kotovn.schemas.dialogue import DialogueFrame
from kotovn.schemas.rag import (
    EpisodeChunk,
    GlossaryEntry,
    KnowledgeSnapshot,
    RAGContext,
    ReflectionProposal,
    RelationEdge,
)


@runtime_checkable
class KnowledgeStore(Protocol):
    """Per-game hybrid knowledge store with snapshot/rollback support."""

    def retrieve(
        self,
        text: str,
        *,
        speaker: str | None = None,
        top_k_episodes: int = 4,
        include_drafts: bool = True,
    ) -> RAGContext: ...

    def upsert_glossary(self, entry: GlossaryEntry) -> GlossaryEntry: ...

    def verify_glossary(self, source_term: str, target_term: str) -> GlossaryEntry: ...

    def upsert_relation(self, edge: RelationEdge) -> RelationEdge: ...

    def add_episode(self, chunk: EpisodeChunk) -> None: ...

    def create_snapshot(self, label: str, route_name: str = "common") -> KnowledgeSnapshot: ...

    def rollback_snapshot(self, snapshot_id: str) -> KnowledgeSnapshot: ...


@runtime_checkable
class Reflector(Protocol):
    """Asynchronous extractor powering 'learn while translating' maintenance."""

    async def reflect(
        self,
        window: list[tuple[DialogueFrame, str]],
    ) -> ReflectionProposal: ...
