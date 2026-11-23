"""End-to-end GPU-free integration tests for the translation + online RAG pipeline."""

from __future__ import annotations

from kotovn.backends.mock import HeuristicReflector, MockLLMBackend, MockVisionBackend
from kotovn.config import AppConfig
from kotovn.persona.registry import FilePersonaRegistry
from kotovn.pipeline.translator import TranslationPipeline, build_translation_prompts
from kotovn.rag.memory_store import InMemoryKnowledgeStore
from kotovn.schemas.dialogue import DialogueFrame
from kotovn.schemas.rag import EntryStatus, GlossaryEntry


async def test_pipeline_translates_with_glossary_persona_and_learns_online(
    cpu_config: AppConfig,
    knowledge_store: InMemoryKnowledgeStore,
    persona_registry: FilePersonaRegistry,
    sample_frames: list[DialogueFrame],
) -> None:
    knowledge_store.verify_glossary("桜ヶ丘学園", "樱丘学园")
    persona_registry.adapt_from_archetype(
        persona_id="chinatsu",
        character_name="千夏",
        archetype_id="tsundere",
    )

    pipeline = TranslationPipeline(
        config=cpu_config,
        llm=MockLLMBackend(),
        knowledge_store=knowledge_store,
        persona_repo=persona_registry,
        reflector=HeuristicReflector(),
    )

    # Frame 1: pre-seeded verified term is matched and injected.
    res1 = await pipeline.translate_frame(sample_frames[0])
    assert "樱丘学园" in res1.translated_text
    assert res1.persona_id == "chinatsu"
    assert [h.target_term for h in res1.glossary_hits] == ["樱丘学园"]

    # Frame 2 triggers automatic background reflection (reflection_interval == 2).
    res2 = await pipeline.translate_frame(sample_frames[1])
    assert res2.telemetry.output_tokens > 0

    # Verify '星屑部' was learned automatically as a DRAFT glossary entry.
    learned = {entry.source_term: entry for entry in knowledge_store.list_glossary()}
    assert "星屑部" in learned
    assert learned["星屑部"].status is EntryStatus.DRAFT

    # User hover-edits the term in HUD -> promoted to VERIFIED.
    verified = knowledge_store.verify_glossary("星屑部", "星屑社")
    assert verified.status is EntryStatus.VERIFIED

    # Subsequent frame immediately uses the verified translation.
    res3 = await pipeline.translate_frame(sample_frames[1])
    assert "星屑社" in res3.translated_text


async def test_mock_vision_backend_replays_frames(sample_frames: list[DialogueFrame]) -> None:
    vision = MockVisionBackend(sample_frames)
    replayed = await vision.extract_dialogue(b"fake-png-bytes", game_id="sakura_demo")
    assert replayed is not None
    assert replayed.frame_id == "sample-001"
    assert replayed.visual_context.speaker_emotion == "blushing"


def test_build_translation_prompts_includes_visual_and_persona_context(
    knowledge_store: InMemoryKnowledgeStore,
    persona_registry: FilePersonaRegistry,
    sample_frames: list[DialogueFrame],
) -> None:
    knowledge_store.upsert_glossary(
        GlossaryEntry(source_term="桜ヶ丘学園", target_term="樱丘学园", status=EntryStatus.VERIFIED)
    )
    persona = persona_registry.adapt_from_archetype(
        persona_id="chinatsu",
        character_name="千夏",
        archetype_id="tsundere",
    )
    rag_ctx = knowledge_store.retrieve(sample_frames[0].raw_text, speaker="千夏")
    sys_prompt, user_prompt = build_translation_prompts(
        sample_frames[0],
        rag_context=rag_ctx,
        persona=persona,
    )

    assert "[GLOSSARY - MUST FOLLOW EXACTLY]" in sys_prompt
    assert "[SPEAKER PERSONA: 千夏]" in sys_prompt
    assert "VISUAL: scene=classroom_sunset, emotion=blushing" in user_prompt
