"""Shared pytest fixtures for GPU-free CI verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kotovn.config import AppConfig, RuntimeMode, build_profile_config
from kotovn.persona.registry import FilePersonaRegistry
from kotovn.rag.memory_store import InMemoryKnowledgeStore
from kotovn.schemas.dialogue import DialogueFrame

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def cpu_config() -> AppConfig:
    cfg = build_profile_config(RuntimeMode.CPU_ONLY, cpu_threads=8)
    cfg.rag.reflection_interval = 2
    return cfg


@pytest.fixture
def knowledge_store() -> InMemoryKnowledgeStore:
    return InMemoryKnowledgeStore()


@pytest.fixture
def persona_registry() -> FilePersonaRegistry:
    return FilePersonaRegistry()


@pytest.fixture
def sample_frames() -> list[DialogueFrame]:
    raw = json.loads((FIXTURES_DIR / "sample_vn_dialogue.json").read_text(encoding="utf-8"))
    return [DialogueFrame.model_validate(item) for item in raw]
