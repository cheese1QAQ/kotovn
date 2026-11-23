"""Hardware-aware runtime configuration for local-first inference.

KotoVN runs strictly on the local machine:
- ``cpu_only``: Zero VRAM usage, designed for CPU inference alongside any game.
- ``hybrid_12gb``: Balanced partial GPU offload capped safely for 12GB VRAM GPUs.
- ``research_gpu``: Higher VRAM budget for local quantization/VLM benchmarks.

Google Colab is reserved strictly for offline lightweighting research under ``research/``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class RuntimeMode(str, Enum):
    """Pre-tuned hardware execution profiles."""

    CPU_ONLY = "cpu_only"
    HYBRID_12GB = "hybrid_12gb"
    RESEARCH_GPU = "research_gpu"


class KVCacheQuantType(str, Enum):
    """Supported llama.cpp KV cache precision levels."""

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


class VisionMode(str, Enum):
    """Pluggable vision frontend modes (OCR/VLM first)."""

    LIGHTWEIGHT_OCR = "lightweight_ocr"
    OCR_PLUS_TRIGGERED_VLM = "ocr_plus_triggered_vlm"
    COMPACT_VLM = "compact_vlm"


class LLMRuntimeConfig(BaseModel):
    """Configuration for the local quantized LLM backend."""

    backend: str = Field(default="mock", description="Backend id: 'mock' or 'llama_cpp'.")
    model_path: Path | None = Field(default=None, description="Path to local GGUF weight file.")
    n_ctx: int = Field(default=4096, ge=512, le=32768)
    n_threads: int = Field(default=8, ge=1, le=128)
    n_gpu_layers: int = Field(
        default=0,
        ge=-1,
        description="0 keeps inference strictly on CPU; >0 offloads N layers; -1 offloads all.",
    )
    kv_cache_type: KVCacheQuantType = Field(default=KVCacheQuantType.Q8_0)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=16, le=2048)


class VisionRuntimeConfig(BaseModel):
    """Configuration for screen capture and OCR/VLM perception."""

    mode: VisionMode = Field(default=VisionMode.LIGHTWEIGHT_OCR)
    poll_fps: float = Field(default=5.0, ge=1.0, le=30.0)
    stability_ms: int = Field(
        default=220,
        ge=50,
        le=2000,
        description="Minimum stable duration before triggering OCR/VLM after typewriter stops.",
    )
    auto_roi: bool = Field(
        default=True,
        description="Automatically detect the dialogue box region for zero-friction UX.",
    )
    vlm_change_threshold: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Scene/portrait perceptual hash delta required to trigger optional VLM.",
    )


class RAGRuntimeConfig(BaseModel):
    """Configuration for Hybrid RAG and asynchronous online learning."""

    top_k_episodes: int = Field(default=4, ge=0, le=20)
    reflection_interval: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of dialogue turns between async background reflection runs.",
    )
    auto_apply_draft_glossary: bool = Field(
        default=True,
        description="Immediately use draft terms learned online while flagging them in HUD.",
    )


class AppConfig(BaseModel):
    """Top-level KotoVN runtime configuration."""

    profile: RuntimeMode = Field(default=RuntimeMode.CPU_ONLY)
    source_language: str = Field(default="ja")
    target_language: str = Field(default="zh-CN")
    workspace_dir: Path = Field(default=Path("data/games/default"))
    llm: LLMRuntimeConfig = Field(default_factory=LLMRuntimeConfig)
    vision: VisionRuntimeConfig = Field(default_factory=VisionRuntimeConfig)
    rag: RAGRuntimeConfig = Field(default_factory=RAGRuntimeConfig)


def build_profile_config(
    mode: RuntimeMode,
    *,
    cpu_threads: int = 8,
    backend: str = "mock",
) -> AppConfig:
    """Build a zero-friction default configuration for the selected hardware profile."""
    safe_threads = max(1, min(cpu_threads, 16))
    if mode is RuntimeMode.CPU_ONLY:
        return AppConfig(
            profile=mode,
            llm=LLMRuntimeConfig(
                backend=backend,
                n_ctx=4096,
                n_threads=safe_threads,
                n_gpu_layers=0,
                kv_cache_type=KVCacheQuantType.Q8_0,
            ),
            vision=VisionRuntimeConfig(mode=VisionMode.LIGHTWEIGHT_OCR),
        )
    if mode is RuntimeMode.HYBRID_12GB:
        return AppConfig(
            profile=mode,
            llm=LLMRuntimeConfig(
                backend=backend,
                n_ctx=4096,
                n_threads=max(1, safe_threads // 2),
                n_gpu_layers=24,
                kv_cache_type=KVCacheQuantType.Q8_0,
            ),
            vision=VisionRuntimeConfig(mode=VisionMode.OCR_PLUS_TRIGGERED_VLM),
        )
    return AppConfig(
        profile=RuntimeMode.RESEARCH_GPU,
        llm=LLMRuntimeConfig(
            backend=backend,
            n_ctx=8192,
            n_threads=max(1, safe_threads // 2),
            n_gpu_layers=-1,
            kv_cache_type=KVCacheQuantType.F16,
        ),
        vision=VisionRuntimeConfig(mode=VisionMode.COMPACT_VLM),
    )


def recommend_profile(*, vram_mb: int, prefer_cpu: bool = True) -> RuntimeMode:
    """Recommend a safe local runtime profile based on detected VRAM and user preference."""
    if prefer_cpu or vram_mb < 8000:
        return RuntimeMode.CPU_ONLY
    if vram_mb <= 14000:
        return RuntimeMode.HYBRID_12GB
    return RuntimeMode.RESEARCH_GPU
