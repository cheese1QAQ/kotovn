"""Tests for cross-novel archetype loading and character persona adaptation."""

from __future__ import annotations

import pytest

from kotovn.persona.registry import FilePersonaRegistry
from kotovn.schemas.persona import PersonaOrigin, StyleAnchor


def test_builtin_archetypes_load_cleanly(persona_registry: FilePersonaRegistry) -> None:
    archetypes = persona_registry.list_archetypes()
    ids = {a.archetype_id for a in archetypes}
    assert {"tsundere", "ojou_sama", "kuudere"}.issubset(ids)


def test_adapt_from_archetype_merges_overrides_and_prioritizes_emotion_anchors(
    persona_registry: FilePersonaRegistry,
) -> None:
    custom_anchor = StyleAnchor(
        source="先輩、今日は屋上で待ってますからね！",
        target="前辈，我今天会在天台等你的哦！",
        emotion_tag="blushing",
    )
    card = persona_registry.adapt_from_archetype(
        persona_id="chinatsu",
        character_name="千夏",
        aliases=["夏酱", "Chinatsu"],
        archetype_id="tsundere",
        pronoun_overrides={"うち": "人家"},
        extra_endings=["真是拿你没办法"],
        extra_anchors=[custom_anchor],
        origin=PersonaOrigin.SCRIPT_ADAPTED,
    )

    assert persona_registry.resolve_persona("夏酱") is not None
    assert card.fingerprint.self_pronoun_map["うち"] == "人家"
    assert card.fingerprint.self_pronoun_map["あたし"] == "人家"
    assert "真是拿你没办法" in card.fingerprint.sentence_endings

    selected = card.select_anchors(emotion="blushing", limit=2)
    assert len(selected) == 2
    assert selected[0].source == custom_anchor.source


def test_unknown_archetype_raises_key_error(persona_registry: FilePersonaRegistry) -> None:
    with pytest.raises(KeyError, match="unknown_type"):
        persona_registry.adapt_from_archetype(
            persona_id="x",
            character_name="X",
            archetype_id="unknown_type",
        )
