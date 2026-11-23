"""In-memory Hybrid RAG reference implementation with Save/Load checkpoint rollback.

Serves as:
1. The exact behavioral contract for Phase 3's SQLite + SQLite-vec/LanceDB backend.
2. A fast, zero-dependency store for CI unit/integration tests and CLI demos.
"""

from __future__ import annotations

import copy
from uuid import uuid4

from kotovn.schemas.rag import (
    EntryStatus,
    EpisodeChunk,
    GlossaryEntry,
    KnowledgeSnapshot,
    RAGContext,
    RelationEdge,
)


def _char_bigrams(text: str) -> set[str]:
    cleaned = "".join(text.split())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


class InMemoryKnowledgeStore:
    """Three-layer hybrid knowledge store (Glossary + Episodes + Relations) with snapshots."""

    def __init__(self) -> None:
        self._glossary: dict[str, GlossaryEntry] = {}
        self._relations: dict[tuple[str, str], RelationEdge] = {}
        self._episodes: list[EpisodeChunk] = []
        self._snapshots: dict[
            str,
            tuple[
                KnowledgeSnapshot,
                dict[str, GlossaryEntry],
                dict[tuple[str, str], RelationEdge],
                list[EpisodeChunk],
            ],
        ] = {}

    def retrieve(
        self,
        text: str,
        *,
        speaker: str | None = None,
        top_k_episodes: int = 4,
        include_drafts: bool = True,
    ) -> RAGContext:
        """Retrieve exact glossary hits, speaker relations, and relevant episode memories."""
        # L1: Exact substring match sorted longest-first to avoid partial shadowing.
        glossary_hits = [
            entry
            for term, entry in sorted(
                self._glossary.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            if term in text and (include_drafts or entry.status is EntryStatus.VERIFIED)
        ]

        # L3: Relation edges involving the current speaker or entities mentioned in text.
        relations: list[RelationEdge] = []
        for edge in self._relations.values():
            if not include_drafts and edge.status is not EntryStatus.VERIFIED:
                continue
            speaker_match = speaker is not None and edge.source_character == speaker
            text_match = edge.target_character in text or edge.source_character in text
            if speaker_match or text_match:
                relations.append(edge)

        # L2: Lightweight character-bigram overlap scoring (stand-in for Dense + BM25).
        episodes: list[EpisodeChunk] = []
        if top_k_episodes > 0 and self._episodes:
            query_grams = _char_bigrams(text)
            scored: list[tuple[float, int, EpisodeChunk]] = []
            for idx, chunk in enumerate(self._episodes):
                chunk_grams = _char_bigrams(chunk.summary)
                overlap = len(query_grams & chunk_grams)
                speaker_bonus = 1.5 if speaker and speaker in chunk.speakers else 0.0
                score = float(overlap) + speaker_bonus
                scored.append((score, idx, chunk))
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            episodes = [chunk for _, _, chunk in scored[:top_k_episodes]]

        return RAGContext(glossary=glossary_hits, relations=relations, episodes=episodes)

    def upsert_glossary(self, entry: GlossaryEntry) -> GlossaryEntry:
        """Insert or update a glossary entry while protecting verified terms from draft clobber."""
        existing = self._glossary.get(entry.source_term)
        if (
            existing is not None
            and existing.status is EntryStatus.VERIFIED
            and entry.status is EntryStatus.DRAFT
        ):
            return existing
        stored = entry.model_copy(deep=True)
        self._glossary[stored.source_term] = stored
        return stored

    def verify_glossary(self, source_term: str, target_term: str) -> GlossaryEntry:
        """Promote a glossary term to VERIFIED (typically triggered from HUD hover-edit)."""
        existing = self._glossary.get(source_term)
        if existing is None:
            entry = GlossaryEntry(
                source_term=source_term,
                target_term=target_term,
                status=EntryStatus.VERIFIED,
            )
        else:
            entry = existing.model_copy(
                update={"target_term": target_term, "status": EntryStatus.VERIFIED}
            )
        self._glossary[source_term] = entry
        return entry

    def upsert_relation(self, edge: RelationEdge) -> RelationEdge:
        """Insert or update a character relation edge without overriding verified links."""
        key = (edge.source_character, edge.target_character)
        existing = self._relations.get(key)
        if (
            existing is not None
            and existing.status is EntryStatus.VERIFIED
            and edge.status is EntryStatus.DRAFT
        ):
            return existing
        stored = edge.model_copy(deep=True)
        self._relations[key] = stored
        return stored

    def add_episode(self, chunk: EpisodeChunk) -> None:
        """Append a compressed dialogue/scene episode chunk."""
        self._episodes.append(chunk.model_copy(deep=True))

    def create_snapshot(self, label: str, route_name: str = "common") -> KnowledgeSnapshot:
        """Capture an immutable checkpoint for VN Save/Load and multi-route branching."""
        snapshot_id = f"snap-{uuid4().hex[:8]}"
        meta = KnowledgeSnapshot(
            snapshot_id=snapshot_id,
            label=label,
            route_name=route_name,
            glossary_count=len(self._glossary),
            relation_count=len(self._relations),
            episode_count=len(self._episodes),
        )
        self._snapshots[snapshot_id] = (
            meta,
            copy.deepcopy(self._glossary),
            copy.deepcopy(self._relations),
            copy.deepcopy(self._episodes),
        )
        return meta

    def rollback_snapshot(self, snapshot_id: str) -> KnowledgeSnapshot:
        """Restore knowledge state cleanly to a previously recorded checkpoint."""
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Unknown snapshot_id: {snapshot_id}")
        meta, glossary, relations, episodes = self._snapshots[snapshot_id]
        self._glossary = copy.deepcopy(glossary)
        self._relations = copy.deepcopy(relations)
        self._episodes = copy.deepcopy(episodes)
        return meta

    def list_glossary(self) -> list[GlossaryEntry]:
        """Return all glossary terms sorted alphabetically by source term."""
        return [self._glossary[k] for k in sorted(self._glossary)]
