"""Persona registry managing cross-novel YAML archetypes and per-game character cards."""

from __future__ import annotations

from pathlib import Path

import yaml

from kotovn.schemas.persona import (
    ArchetypeTemplate,
    LinguisticFingerprint,
    PersonaCard,
    PersonaOrigin,
    StyleAnchor,
)

DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class FilePersonaRegistry:
    """Loads pre-built YAML archetypes and builds adapted character PersonaCards."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        self._templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self._archetypes: dict[str, ArchetypeTemplate] = {}
        self._personas: dict[str, PersonaCard] = {}
        self._alias_index: dict[str, str] = {}
        self._load_archetypes()

    def _load_archetypes(self) -> None:
        if not self._templates_dir.exists():
            return
        for yaml_file in sorted(self._templates_dir.glob("*.yaml")):
            raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            template = ArchetypeTemplate.model_validate(raw)
            self._archetypes[template.archetype_id] = template

    def list_archetypes(self) -> list[ArchetypeTemplate]:
        return [self._archetypes[k] for k in sorted(self._archetypes)]

    def get_archetype(self, archetype_id: str) -> ArchetypeTemplate | None:
        return self._archetypes.get(archetype_id)

    def upsert_persona(self, card: PersonaCard) -> PersonaCard:
        stored = card.model_copy(deep=True)
        self._personas[stored.persona_id] = stored
        self._alias_index[stored.character_name] = stored.persona_id
        for alias in stored.aliases:
            self._alias_index[alias] = stored.persona_id
        return stored

    def resolve_persona(self, speaker_name: str | None) -> PersonaCard | None:
        if not speaker_name:
            return None
        cleaned = speaker_name.strip()
        persona_id = self._alias_index.get(cleaned)
        if persona_id is None:
            return None
        return self._personas.get(persona_id)

    def adapt_from_archetype(
        self,
        *,
        persona_id: str,
        character_name: str,
        archetype_id: str,
        aliases: list[str] | None = None,
        summary_override: str | None = None,
        pronoun_overrides: dict[str, str] | None = None,
        extra_endings: list[str] | None = None,
        extra_anchors: list[StyleAnchor] | None = None,
        origin: PersonaOrigin = PersonaOrigin.AUTO_INFERRED,
    ) -> PersonaCard:
        """Create a character PersonaCard seeded from a cross-novel archetype plus overrides."""
        template = self._archetypes.get(archetype_id)
        if template is None:
            raise KeyError(f"Unknown archetype_id: {archetype_id}")

        merged_pronouns = dict(template.fingerprint.self_pronoun_map)
        if pronoun_overrides:
            merged_pronouns.update(pronoun_overrides)

        merged_endings = list(template.fingerprint.sentence_endings)
        for ending in extra_endings or []:
            if ending not in merged_endings:
                merged_endings.append(ending)

        fingerprint = LinguisticFingerprint(
            self_pronoun_map=merged_pronouns,
            sentence_endings=merged_endings,
            forbidden_expressions=list(template.fingerprint.forbidden_expressions),
            politeness_level=template.fingerprint.politeness_level,
            punctuation_style=template.fingerprint.punctuation_style,
        )
        # Put character-specific anchors first so they take priority over generic archetype anchors.
        anchors = list(extra_anchors or []) + list(template.style_anchors)

        card = PersonaCard(
            persona_id=persona_id,
            character_name=character_name,
            aliases=list(aliases or []),
            archetype_base=archetype_id,
            origin=origin,
            summary=summary_override or template.description,
            fingerprint=fingerprint,
            style_anchors=anchors,
        )
        return self.upsert_persona(card)
