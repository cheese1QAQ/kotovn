"""CLI utilities for local hardware diagnostics and offline pipeline demos."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys

from kotovn import __version__
from kotovn.backends.mock import HeuristicReflector, MockLLMBackend
from kotovn.config import RuntimeMode, build_profile_config, recommend_profile
from kotovn.persona.registry import FilePersonaRegistry
from kotovn.pipeline.translator import TranslationPipeline
from kotovn.rag.memory_store import InMemoryKnowledgeStore
from kotovn.schemas.dialogue import DialogueFrame, VisualContext
from kotovn.schemas.rag import EntryStatus, GlossaryEntry


def _detect_vram_mb() -> int:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        first_line = output.strip().splitlines()[0]
        return int(first_line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return 0


def cmd_doctor() -> int:
    cpu_threads = os.cpu_count() or 4
    vram_mb = _detect_vram_mb()
    cpu_rec = recommend_profile(vram_mb=vram_mb, prefer_cpu=True)
    gpu_rec = recommend_profile(vram_mb=vram_mb, prefer_cpu=False)

    print(f"KotoVN v{__version__} — Local Runtime Diagnostics")
    print(f"  CPU Logical Threads : {cpu_threads}")
    print(f"  Detected GPU VRAM   : {vram_mb} MiB" if vram_mb else "  Detected GPU VRAM   : None")
    print(f"  Default Recommendation (CPU-first) : {cpu_rec.value}")
    print(f"  Optional GPU-assisted Profile      : {gpu_rec.value}")
    print("  Colab Policy                       : Offline lightweighting research ONLY")
    return 0


async def _run_demo() -> int:
    cfg = build_profile_config(RuntimeMode.CPU_ONLY, cpu_threads=os.cpu_count() or 8)
    cfg.rag.reflection_interval = 2

    store = InMemoryKnowledgeStore()
    store.upsert_glossary(
        GlossaryEntry(
            source_term="桜ヶ丘学園",
            target_term="樱丘学园",
            category="location",
            status=EntryStatus.VERIFIED,
        )
    )
    registry = FilePersonaRegistry()
    registry.adapt_from_archetype(
        persona_id="chinatsu",
        character_name="千夏",
        archetype_id="tsundere",
    )

    pipeline = TranslationPipeline(
        config=cfg,
        llm=MockLLMBackend(),
        knowledge_store=store,
        persona_repo=registry,
        reflector=HeuristicReflector(),
    )

    frames = [
        DialogueFrame(
            frame_id="f-001",
            speaker_raw="千夏",
            raw_text="べ、別に桜ヶ丘学園の「星屑部」に誘いたいわけじゃないんだからね！",
            visual_context=VisualContext(scene_tag="classroom_sunset", speaker_emotion="blushing"),
        ),
        DialogueFrame(
            frame_id="f-002",
            speaker_raw="千夏",
            raw_text="とにかく「星屑部」の部室に来なさいよ！",
            visual_context=VisualContext(scene_tag="classroom_sunset", speaker_emotion="angry"),
        ),
    ]

    for frame in frames:
        result = await pipeline.translate_frame(frame)
        hits = [f"{h.source_term}->{h.target_term}({h.status.value})" for h in result.glossary_hits]
        print(f"[{result.frame_id}] ({result.speaker_resolved} | persona={result.persona_id})")
        print(f"  SRC : {result.source_text}")
        print(f"  DST : {result.translated_text}")
        print(f"  RAG : {hits or 'none'}")

    print("\nLearned glossary after online reflection:")
    for item in store.list_glossary():
        print(f"  - {item.source_term} => {item.target_term} [{item.status.value}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kotovn", description="KotoVN local VN translator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect local CPU/VRAM and recommend runtime profile")
    sub.add_parser("demo", help="Run an offline mock translation + online RAG demo")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return cmd_doctor()
    if args.command == "demo":
        return asyncio.run(_run_demo())
    return 1


if __name__ == "__main__":
    sys.exit(main())
