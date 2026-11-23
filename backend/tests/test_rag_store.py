"""Tests for Hybrid RAG retrieval, verified-term protection, and SL checkpoint rollback."""

from __future__ import annotations

import pytest

from kotovn.rag.memory_store import InMemoryKnowledgeStore
from kotovn.schemas.rag import EntryStatus, EpisodeChunk, GlossaryEntry, RelationEdge


def test_verified_glossary_is_protected_from_draft_overwrite(
    knowledge_store: InMemoryKnowledgeStore,
) -> None:
    knowledge_store.verify_glossary("桜ヶ丘学園", "樱丘学园")
    stored = knowledge_store.upsert_glossary(
        GlossaryEntry(
            source_term="桜ヶ丘学園",
            target_term="樱花山学院",
            status=EntryStatus.DRAFT,
        )
    )
    assert stored.target_term == "樱丘学园"
    assert stored.status is EntryStatus.VERIFIED


def test_hybrid_retrieve_matches_glossary_relations_and_episodes(
    knowledge_store: InMemoryKnowledgeStore,
) -> None:
    knowledge_store.verify_glossary("星屑部", "星屑社")
    knowledge_store.upsert_relation(
        RelationEdge(
            source_character="千夏",
            target_character="悠真",
            addressing_term="笨蛋前辈",
            relation_label="club_senior",
            status=EntryStatus.VERIFIED,
        )
    )
    knowledge_store.add_episode(
        EpisodeChunk(
            chunk_id="ep-1",
            summary="千夏在黄昏教室里把悠真拉进了星屑社。",
            speakers=["千夏", "悠真"],
            scene_tag="classroom_sunset",
        )
    )

    ctx = knowledge_store.retrieve("星屑部の活動について", speaker="千夏", top_k_episodes=2)
    assert [g.target_term for g in ctx.glossary] == ["星屑社"]
    assert len(ctx.relations) == 1
    assert ctx.relations[0].addressing_term == "笨蛋前辈"
    assert len(ctx.episodes) == 1


def test_checkpoint_snapshot_and_rollback_prevents_route_contamination(
    knowledge_store: InMemoryKnowledgeStore,
) -> None:
    knowledge_store.verify_glossary("桜ヶ丘学園", "樱丘学园")
    snap = knowledge_store.create_snapshot(label="before_route_split", route_name="common")

    # Simulate playing Route A and learning Route-A-only knowledge.
    knowledge_store.upsert_glossary(
        GlossaryEntry(source_term="秘密の鍵", target_term="秘密之钥", status=EntryStatus.DRAFT)
    )
    assert len(knowledge_store.list_glossary()) == 2

    # Roll back when player loads an earlier save to enter Route B.
    restored = knowledge_store.rollback_snapshot(snap.snapshot_id)
    assert restored.label == "before_route_split"
    assert [g.source_term for g in knowledge_store.list_glossary()] == ["桜ヶ丘学園"]


def test_rollback_unknown_snapshot_raises(knowledge_store: InMemoryKnowledgeStore) -> None:
    with pytest.raises(KeyError, match="snap-missing"):
        knowledge_store.rollback_snapshot("snap-missing")
