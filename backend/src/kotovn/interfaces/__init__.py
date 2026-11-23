"""Abstract protocol boundaries keeping AI backends and UI strictly decoupled."""

from __future__ import annotations

from kotovn.interfaces.llm import LLMBackend, LLMGenerationConfig, LLMResponse
from kotovn.interfaces.persona import PersonaRepository
from kotovn.interfaces.rag import KnowledgeStore, Reflector
from kotovn.interfaces.script import ScriptLine, ScriptSource
from kotovn.interfaces.vision import VisionBackend

__all__ = [
    "KnowledgeStore",
    "LLMBackend",
    "LLMGenerationConfig",
    "LLMResponse",
    "PersonaRepository",
    "Reflector",
    "ScriptLine",
    "ScriptSource",
    "VisionBackend",
]
