"""Deterministic GPU-free test doubles for CI pipelines and rapid UI iteration."""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator

from kotovn.interfaces.llm import LLMGenerationConfig, LLMResponse
from kotovn.schemas.dialogue import BoundingBox, DialogueFrame, SourceType, VisualContext
from kotovn.schemas.rag import EntryStatus, EpisodeChunk, GlossaryEntry, ReflectionProposal

_QUOTED_TERM_RE = re.compile(r"「([^「」]{2,12})」")


class MockLLMBackend:
    """Deterministic LLM backend honoring glossary hints for repeatable CI tests."""

    def __init__(
        self,
        *,
        model_name: str = "mock-vn-q4_k_m",
        canned_responses: dict[str, str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._canned = canned_responses or {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def _synthesize(self, system_prompt: str, user_prompt: str) -> str:
        for key, translation in self._canned.items():
            if key in user_prompt:
                return translation

        # Extract the raw source text line from the structured user prompt if present.
        source = user_prompt.strip()
        for line in user_prompt.splitlines():
            if line.startswith("SOURCE: "):
                source = line.removeprefix("SOURCE: ").strip()
                break

        # Apply any glossary pairs listed in the system prompt ("src => dst").
        translated = source
        for line in system_prompt.splitlines():
            if " => " in line and line.strip().startswith("- "):
                raw_pair = line.strip().removeprefix("- ")
                src_term, _, rest = raw_pair.partition(" => ")
                dst_term = rest.split(" (", maxsplit=1)[0].strip()
                if src_term and dst_term:
                    translated = translated.replace(src_term.strip(), dst_term)

        return f"[MOCK译] {translated}"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        _ = config
        start = time.perf_counter()
        text = self._synthesize(system_prompt, user_prompt)
        elapsed_ms = max(0.5, (time.perf_counter() - start) * 1000.0)
        tokens = max(1, len(text) // 2)
        return LLMResponse(
            text=text,
            output_tokens=tokens,
            ttft_ms=round(elapsed_ms * 0.4, 3),
            total_ms=round(elapsed_ms, 3),
        )

    async def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMGenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        response = await self.generate(system_prompt, user_prompt, config=config)
        chunk_size = 4
        for idx in range(0, len(response.text), chunk_size):
            yield response.text[idx : idx + chunk_size]


class MockVisionBackend:
    """Replays pre-recorded DialogueFrames without requiring live screen capture or GPUs."""

    def __init__(self, frames: list[DialogueFrame] | None = None) -> None:
        self._frames = list(frames or [])
        self._cursor = 0

    @property
    def backend_name(self) -> str:
        return "mock-vision-replay"

    async def extract_dialogue(
        self,
        image_bytes: bytes,
        *,
        game_id: str,
        roi_box: BoundingBox | None = None,
    ) -> DialogueFrame | None:
        if not image_bytes:
            return None
        if self._cursor < len(self._frames):
            frame = self._frames[self._cursor]
            self._cursor += 1
            return frame.model_copy(update={"game_id": game_id, "roi_box": roi_box})
        decoded = image_bytes.decode("utf-8", errors="ignore").strip()
        if not decoded:
            return None
        return DialogueFrame(
            frame_id=f"mock-frame-{self._cursor}",
            game_id=game_id,
            source_type=SourceType.OCR,
            raw_text=decoded,
            visual_context=VisualContext(scene_tag="mock_scene"),
            roi_box=roi_box,
        )


class HeuristicReflector:
    """Rule-based async reflector used in tests and ultra-lightweight CPU mode.

    Extracts Japanese bracketed proper nouns (「...」) as draft glossary candidates
    and compresses recent dialogue turns into an episodic memory summary.
    """

    async def reflect(
        self,
        window: list[tuple[DialogueFrame, str]],
    ) -> ReflectionProposal:
        if not window:
            return ReflectionProposal()

        candidates: dict[str, GlossaryEntry] = {}
        speakers: list[str] = []
        snippets: list[str] = []
        last_scene: str | None = None

        for frame, translated in window:
            if frame.speaker_raw and frame.speaker_raw not in speakers:
                speakers.append(frame.speaker_raw)
            if frame.visual_context.scene_tag:
                last_scene = frame.visual_context.scene_tag
            snippets.append(translated)

            for term in _QUOTED_TERM_RE.findall(frame.raw_text):
                if term not in candidates:
                    candidates[term] = GlossaryEntry(
                        source_term=term,
                        target_term=term,
                        category="term",
                        status=EntryStatus.DRAFT,
                        note="Auto-discovered from dialogue quotation marks.",
                        evidence_frame_id=frame.frame_id,
                    )

        first_id = window[0][0].frame_id
        last_id = window[-1][0].frame_id
        summary = EpisodeChunk(
            chunk_id=f"ep-{first_id}-{last_id}",
            summary=" / ".join(snippets[-3:]),
            speakers=speakers,
            scene_tag=last_scene,
        )
        return ReflectionProposal(
            glossary_candidates=list(candidates.values()),
            episode_summary=summary,
        )
