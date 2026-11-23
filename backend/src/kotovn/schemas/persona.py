"""Schemas for character persona templates (cross-novel archetypes + script/online adaptation)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PersonaOrigin(str, Enum):
    """How a character's persona card was constructed."""

    ARCHETYPE = "archetype"
    AUTO_INFERRED = "auto_inferred"
    SCRIPT_ADAPTED = "script_adapted"
    USER_VERIFIED = "user_verified"


class LinguisticFingerprint(BaseModel):
    """Compact stylistic rules tailored for small quantized LLMs (3B-8B)."""

    self_pronoun_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps source first-person pronouns to target voice, e.g., {'わたくし': '本小姐'}."
        ),
    )
    sentence_endings: list[str] = Field(
        default_factory=list,
        description="Characteristic Chinese sentence particles/endings to favor.",
    )
    forbidden_expressions: list[str] = Field(
        default_factory=list,
        description="Words or slang that break character.",
    )
    politeness_level: int = Field(
        default=3,
        ge=1,
        le=5,
        description="1=very rough/blunt, 3=neutral casual, 5=aristocratic/highly formal.",
    )
    punctuation_style: str = Field(
        default="balanced",
        description="e.g., 'short_ellipsis', 'exclamatory', 'measured_elegant'.",
    )


class StyleAnchor(BaseModel):
    """High-leverage few-shot dialogue pair anchoring character voice."""

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    emotion_tag: str = Field(default="neutral")


class ArchetypeTemplate(BaseModel):
    """Reusable persona archetype pre-built from other classic Visual Novels."""

    archetype_id: str = Field(min_length=1)
    display_name: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    fingerprint: LinguisticFingerprint = Field(default_factory=LinguisticFingerprint)
    style_anchors: list[StyleAnchor] = Field(default_factory=list)


class PersonaCard(BaseModel):
    """Concrete character persona active in the current game workspace."""

    persona_id: str = Field(min_length=1)
    character_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    archetype_base: str | None = Field(default=None)
    origin: PersonaOrigin = Field(default=PersonaOrigin.AUTO_INFERRED)
    summary: str = Field(default="")
    fingerprint: LinguisticFingerprint = Field(default_factory=LinguisticFingerprint)
    style_anchors: list[StyleAnchor] = Field(default_factory=list)

    def select_anchors(self, emotion: str | None = None, limit: int = 3) -> list[StyleAnchor]:
        """Prioritize few-shot anchors matching the detected visual/contextual emotion."""
        if not self.style_anchors or limit <= 0:
            return []
        if emotion is None:
            return self.style_anchors[:limit]
        matched = [a for a in self.style_anchors if a.emotion_tag.lower() == emotion.lower()]
        fallback = [a for a in self.style_anchors if a.emotion_tag.lower() != emotion.lower()]
        return (matched + fallback)[:limit]
