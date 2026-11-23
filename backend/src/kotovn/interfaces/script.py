"""Reserved extension protocol for future Visual Novel script unpackers/importers.

Phase 1-2 operates strictly via OCR/VLM screen capture. When script unpacking
(e.g., Ren'Py, Kirikiri, Artemis, or generic JSON/CSV scripts) is added later,
implementing ``ScriptSource`` unlocks batch persona calibration and glossary
pre-seeding without changing any downstream pipeline code.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ScriptLine(BaseModel):
    """Single parsed dialogue line from an unpacked VN script."""

    line_id: str
    speaker: str | None = None
    text: str = Field(min_length=1)
    scene_id: str | None = None
    route_name: str = "common"


@runtime_checkable
class ScriptSource(Protocol):
    """Extension interface for future VN engine script parsers."""

    @property
    def engine_name(self) -> str: ...

    def iter_lines(self, script_path: Path) -> Iterator[ScriptLine]: ...
