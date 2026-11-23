"""Dialogue perception and translation output schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from kotovn.schemas.rag import EntryStatus


class SourceType(str, Enum):
    """Origin of a captured dialogue line (OCR/VLM now; script unpacking later)."""

    OCR = "ocr"
    VLM = "vlm"
    SCRIPT = "script"


class BoundingBox(BaseModel):
    """Normalized [0, 1] region of interest within the target game window."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class VisualContext(BaseModel):
    """Compact visual signals extracted from the game frame alongside dialogue text."""

    scene_tag: str | None = Field(default=None, description="e.g., 'classroom_sunset'")
    speaker_emotion: str | None = Field(default=None, description="e.g., 'blushing', 'angry'")
    scene_changed: bool = Field(default=False)
    visual_tokens_used: int = Field(default=0, ge=0)


class DialogueFrame(BaseModel):
    """Unified dialogue unit consumed by the translation pipeline."""

    frame_id: str
    game_id: str = Field(default="default")
    source_type: SourceType = Field(default=SourceType.OCR)
    raw_text: str = Field(min_length=1)
    speaker_raw: str | None = Field(default=None)
    listener_hint: str | None = Field(default=None)
    visual_context: VisualContext = Field(default_factory=VisualContext)
    roi_box: BoundingBox | None = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GlossaryHit(BaseModel):
    """Glossary term matched in the current line for HUD highlighting and click-to-edit."""

    source_term: str
    target_term: str
    category: str
    status: EntryStatus
    note: str = ""


class PipelineTelemetry(BaseModel):
    """Latency and throughput breakdown for local quantization and UX profiling."""

    vision_ms: float = Field(default=0.0, ge=0.0)
    rag_ms: float = Field(default=0.0, ge=0.0)
    prompt_ms: float = Field(default=0.0, ge=0.0)
    ttft_ms: float = Field(default=0.0, ge=0.0)
    llm_total_ms: float = Field(default=0.0, ge=0.0)
    output_tokens: int = Field(default=0, ge=0)
    decode_tps: float = Field(default=0.0, ge=0.0)


class TranslationResult(BaseModel):
    """Complete translation output sent to the Overlay HUD and Studio Drawer."""

    frame_id: str
    source_text: str
    translated_text: str
    speaker_resolved: str | None = None
    persona_id: str | None = None
    glossary_hits: list[GlossaryHit] = Field(default_factory=list)
    telemetry: PipelineTelemetry = Field(default_factory=PipelineTelemetry)
