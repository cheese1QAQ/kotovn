"""Tests for hardware-aware local runtime configuration profiles."""

from __future__ import annotations

from kotovn.config import (
    KVCacheQuantType,
    RuntimeMode,
    VisionMode,
    build_profile_config,
    recommend_profile,
)


def test_recommend_profile_defaults_to_cpu_when_preferred() -> None:
    assert recommend_profile(vram_mb=12227, prefer_cpu=True) is RuntimeMode.CPU_ONLY
    assert recommend_profile(vram_mb=4096, prefer_cpu=False) is RuntimeMode.CPU_ONLY


def test_recommend_profile_selects_hybrid_for_12gb_gpu() -> None:
    assert recommend_profile(vram_mb=12227, prefer_cpu=False) is RuntimeMode.HYBRID_12GB
    assert recommend_profile(vram_mb=24576, prefer_cpu=False) is RuntimeMode.RESEARCH_GPU


def test_build_cpu_only_profile_uses_zero_gpu_layers() -> None:
    cfg = build_profile_config(RuntimeMode.CPU_ONLY, cpu_threads=16)
    assert cfg.llm.n_gpu_layers == 0
    assert cfg.llm.n_threads == 16
    assert cfg.llm.kv_cache_type is KVCacheQuantType.Q8_0
    assert cfg.vision.mode is VisionMode.LIGHTWEIGHT_OCR


def test_build_hybrid_12gb_profile_caps_gpu_offload() -> None:
    cfg = build_profile_config(RuntimeMode.HYBRID_12GB, cpu_threads=16)
    assert cfg.llm.n_gpu_layers > 0
    assert cfg.vision.mode is VisionMode.OCR_PLUS_TRIGGERED_VLM
