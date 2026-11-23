"""Protocol for lightweight OCR and compact VLM vision frontends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kotovn.schemas.dialogue import BoundingBox, DialogueFrame


@runtime_checkable
class VisionBackend(Protocol):
    """Extracts structured dialogue and optional visual cues from raw frame bytes."""

    @property
    def backend_name(self) -> str: ...

    async def extract_dialogue(
        self,
        image_bytes: bytes,
        *,
        game_id: str,
        roi_box: BoundingBox | None = None,
    ) -> DialogueFrame | None:
        """Return a DialogueFrame when stable new text is detected, else None."""
        ...
