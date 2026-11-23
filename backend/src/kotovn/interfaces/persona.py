"""Protocol for managing archetype templates and character persona cards."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kotovn.schemas.persona import ArchetypeTemplate, PersonaCard


@runtime_checkable
class PersonaRepository(Protocol):
    """Loads reusable cross-novel archetypes and resolves active game personas."""

    def list_archetypes(self) -> list[ArchetypeTemplate]: ...

    def get_archetype(self, archetype_id: str) -> ArchetypeTemplate | None: ...

    def resolve_persona(self, speaker_name: str | None) -> PersonaCard | None: ...

    def upsert_persona(self, card: PersonaCard) -> PersonaCard: ...
