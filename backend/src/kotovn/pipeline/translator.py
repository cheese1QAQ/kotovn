"""Hot-path translation orchestrator with asynchronous 'learn while translating' reflection."""

from __future__ import annotations

import time

from kotovn.config import AppConfig
from kotovn.interfaces.llm import LLMBackend, LLMGenerationConfig
from kotovn.interfaces.persona import PersonaRepository
from kotovn.interfaces.rag import KnowledgeStore, Reflector
from kotovn.schemas.dialogue import (
    DialogueFrame,
    GlossaryHit,
    PipelineTelemetry,
    TranslationResult,
)
from kotovn.schemas.persona import PersonaCard
from kotovn.schemas.rag import RAGContext, ReflectionProposal


def build_translation_prompts(
    frame: DialogueFrame,
    *,
    rag_context: RAGContext,
    persona: PersonaCard | None,
    source_language: str = "ja",
    target_language: str = "zh-CN",
) -> tuple[str, str]:
    """Assemble compact, quantization-friendly system and user prompts."""
    system_lines: list[str] = [
        f"You are an expert Visual Novel translator ({source_language} -> {target_language}).",
        "Translate faithfully into natural, vivid dialogue. Output ONLY the translated line.",
    ]

    if rag_context.glossary:
        system_lines.append("\n[GLOSSARY - MUST FOLLOW EXACTLY]")
        for entry in rag_context.glossary:
            system_lines.append(
                f"- {entry.source_term} => {entry.target_term} "
                f"({entry.category}, {entry.status.value})"
            )

    if rag_context.relations:
        system_lines.append("\n[CHARACTER RELATIONS]")
        for rel in rag_context.relations:
            system_lines.append(
                f"- {rel.source_character} -> {rel.target_character}: "
                f"calls them '{rel.addressing_term}' ({rel.relation_label}, {rel.politeness})"
            )

    if rag_context.episodes:
        system_lines.append("\n[RECENT STORY CONTEXT]")
        for ep in rag_context.episodes:
            system_lines.append(f"- {ep.summary}")

    if persona is not None:
        fp = persona.fingerprint
        system_lines.append(f"\n[SPEAKER PERSONA: {persona.character_name}]")
        if persona.summary:
            system_lines.append(f"Profile: {persona.summary}")
        if fp.self_pronoun_map:
            pronouns = ", ".join(f"{k}->{v}" for k, v in fp.self_pronoun_map.items())
            system_lines.append(f"First-person style: {pronouns}")
        if fp.sentence_endings:
            system_lines.append(f"Preferred tone/endings: {' / '.join(fp.sentence_endings)}")
        if fp.forbidden_expressions:
            system_lines.append(f"Avoid expressions: {' / '.join(fp.forbidden_expressions)}")

        anchors = persona.select_anchors(emotion=frame.visual_context.speaker_emotion, limit=3)
        if anchors:
            system_lines.append("Style examples:")
            for anchor in anchors:
                system_lines.append(f"  * {anchor.source} => {anchor.target}")

    user_lines: list[str] = []
    if frame.speaker_raw:
        user_lines.append(f"SPEAKER: {frame.speaker_raw}")
    if frame.visual_context.scene_tag or frame.visual_context.speaker_emotion:
        user_lines.append(
            f"VISUAL: scene={frame.visual_context.scene_tag or 'unknown'}, "
            f"emotion={frame.visual_context.speaker_emotion or 'neutral'}"
        )
    user_lines.append(f"SOURCE: {frame.raw_text}")

    return "\n".join(system_lines), "\n".join(user_lines)


class TranslationPipeline:
    """Coordinates foreground low-latency translation and background RAG maintenance."""

    def __init__(
        self,
        *,
        config: AppConfig,
        llm: LLMBackend,
        knowledge_store: KnowledgeStore,
        persona_repo: PersonaRepository,
        reflector: Reflector | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._store = knowledge_store
        self._personas = persona_repo
        self._reflector = reflector
        self._pending_window: list[tuple[DialogueFrame, str]] = []

    async def translate_frame(
        self,
        frame: DialogueFrame,
        *,
        vision_ms: float = 0.0,
    ) -> TranslationResult:
        """Execute the foreground hot path and buffer turns for async reflection."""
        t_rag = time.perf_counter()
        rag_context = self._store.retrieve(
            frame.raw_text,
            speaker=frame.speaker_raw,
            top_k_episodes=self._config.rag.top_k_episodes,
            include_drafts=self._config.rag.auto_apply_draft_glossary,
        )
        rag_ms = (time.perf_counter() - t_rag) * 1000.0

        t_prompt = time.perf_counter()
        persona = self._personas.resolve_persona(frame.speaker_raw)
        system_prompt, user_prompt = build_translation_prompts(
            frame,
            rag_context=rag_context,
            persona=persona,
            source_language=self._config.source_language,
            target_language=self._config.target_language,
        )
        prompt_ms = (time.perf_counter() - t_prompt) * 1000.0

        gen_cfg = LLMGenerationConfig(
            temperature=self._config.llm.temperature,
            max_tokens=self._config.llm.max_tokens,
        )
        llm_resp = await self._llm.generate(system_prompt, user_prompt, config=gen_cfg)

        glossary_hits = [
            GlossaryHit(
                source_term=entry.source_term,
                target_term=entry.target_term,
                category=entry.category,
                status=entry.status,
                note=entry.note,
            )
            for entry in rag_context.glossary
        ]

        telemetry = PipelineTelemetry(
            vision_ms=round(vision_ms, 3),
            rag_ms=round(rag_ms, 3),
            prompt_ms=round(prompt_ms, 3),
            ttft_ms=llm_resp.ttft_ms,
            llm_total_ms=llm_resp.total_ms,
            output_tokens=llm_resp.output_tokens,
            decode_tps=llm_resp.decode_tps,
        )

        result = TranslationResult(
            frame_id=frame.frame_id,
            source_text=frame.raw_text,
            translated_text=llm_resp.text,
            speaker_resolved=persona.character_name if persona else frame.speaker_raw,
            persona_id=persona.persona_id if persona else None,
            glossary_hits=glossary_hits,
            telemetry=telemetry,
        )

        self._pending_window.append((frame, result.translated_text))
        if (
            self._reflector is not None
            and len(self._pending_window) >= self._config.rag.reflection_interval
        ):
            await self.flush_reflection()

        return result

    async def flush_reflection(self) -> ReflectionProposal | None:
        """Run the background Reflector to apply newly learned terms, relations, and episodes."""
        if self._reflector is None or not self._pending_window:
            return None
        batch = list(self._pending_window)
        self._pending_window.clear()

        proposal = await self._reflector.reflect(batch)
        for candidate in proposal.glossary_candidates:
            self._store.upsert_glossary(candidate)
        for relation in proposal.relation_updates:
            self._store.upsert_relation(relation)
        if proposal.episode_summary is not None:
            self._store.add_episode(proposal.episode_summary)
        return proposal
