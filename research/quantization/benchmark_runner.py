#!/usr/bin/env python3
"""Standardized benchmark harness for KotoVN LLM quantization research.

Measures latency (TTFT, decode tok/s), memory/VRAM footprint, and VN-specific
translation quality signals (glossary adherence, non-empty formatting) across
CPU-only and 12GB hybrid GPU offload configurations.

Usage:
    # Validate dataset & pipeline wiring in CI (zero GPU / zero weights):
    python3 research/quantization/benchmark_runner.py --dry-run

    # Benchmark a real quantized GGUF artifact locally:
    python3 research/quantization/benchmark_runner.py \
        --backend llama_cpp --model ./model-Q4_K_M.gguf --n-gpu-layers 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path(__file__).resolve().parent / "eval_dataset.jsonl"


@dataclass
class EvalSample:
    """Single VN benchmark example loaded from eval_dataset.jsonl."""

    id: str
    category: str
    speaker: str
    archetype: str
    emotion: str
    glossary: dict[str, str]
    source: str
    reference: str


@dataclass
class SampleMetric:
    sample_id: str
    category: str
    ttft_ms: float
    total_ms: float
    output_tokens: int
    decode_tps: float
    glossary_expected: int
    glossary_matched: int
    output_preview: str


def load_dataset(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        data = json.loads(line)
        required = {
            "id",
            "category",
            "speaker",
            "archetype",
            "emotion",
            "glossary",
            "source",
            "reference",
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"{path}:{line_no} missing required fields: {sorted(missing)}")
        samples.append(EvalSample(**data))
    if not samples:
        raise ValueError(f"Dataset {path} is empty")
    return samples


def build_eval_prompt(sample: EvalSample) -> tuple[str, str]:
    glossary_lines = "\n".join(f"- {src} => {dst}" for src, dst in sample.glossary.items())
    system = (
        "You are a Visual Novel translator (ja -> zh-CN). "
        "Preserve character voice and strictly follow the glossary. "
        "Output ONLY the Chinese translation.\n"
        f"[GLOSSARY]\n{glossary_lines}\n"
        f"[PERSONA] speaker={sample.speaker}, "
        f"archetype={sample.archetype}, emotion={sample.emotion}"
    )
    user = f"SOURCE: {sample.source}"
    return system, user


def _query_vram_used_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return int(out.strip().splitlines()[0].strip())
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def _run_dry_sample(sample: EvalSample) -> tuple[str, float, float, int]:
    t0 = time.perf_counter()
    translated = sample.source
    for src, dst in sample.glossary.items():
        translated = translated.replace(src, dst)
    text = f"[DRY-RUN] {translated}"
    elapsed_ms = max(0.2, (time.perf_counter() - t0) * 1000.0)
    tokens = max(1, len(text) // 2)
    return text, round(elapsed_ms * 0.4, 3), round(elapsed_ms, 3), tokens


def _run_llama_cpp_sample(
    llm: Any,
    sample: EvalSample,
    max_tokens: int,
) -> tuple[str, float, float, int]:
    system, user = build_eval_prompt(sample)
    t_start = time.perf_counter()
    t_first: float | None = None
    chunks: list[str] = []
    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        stream=True,
    )
    for event in stream:
        delta = event["choices"][0].get("delta", {}).get("content", "")
        if delta:
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(delta)
    t_end = time.perf_counter()

    text = "".join(chunks).strip()
    total_ms = (t_end - t_start) * 1000.0
    ttft_ms = ((t_first - t_start) * 1000.0) if t_first is not None else total_ms
    tokens = max(1, len(chunks))
    return text, round(ttft_ms, 2), round(total_ms, 2), tokens


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_dataset(Path(args.dataset))
    llm: Any = None
    model_size_mb = 0.0

    if not args.dry_run:
        if not args.model:
            raise SystemExit("--model is required unless --dry-run is set.")
        model_path = Path(args.model)
        if not model_path.exists():
            raise SystemExit(f"Model file not found: {model_path}")
        model_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise SystemExit(
                "llama-cpp-python is not installed. Install with: uv sync --extra local-llm"
            ) from exc

        llm = Llama(
            model_path=str(model_path),
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
        )

    metrics: list[SampleMetric] = []
    for sample in samples:
        if args.dry_run:
            output, ttft_ms, total_ms, out_tokens = _run_dry_sample(sample)
        else:
            output, ttft_ms, total_ms, out_tokens = _run_llama_cpp_sample(
                llm, sample, max_tokens=args.max_tokens
            )

        expected = len(sample.glossary)
        matched = sum(1 for dst in sample.glossary.values() if dst in output)
        tps = round(out_tokens / (total_ms / 1000.0), 2) if total_ms > 0 else 0.0
        metrics.append(
            SampleMetric(
                sample_id=sample.id,
                category=sample.category,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                output_tokens=out_tokens,
                decode_tps=tps,
                glossary_expected=expected,
                glossary_matched=matched,
                output_preview=output[:80],
            )
        )

    total_expected = sum(m.glossary_expected for m in metrics)
    total_matched = sum(m.glossary_matched for m in metrics)
    glossary_hit_rate = round(total_matched / total_expected, 4) if total_expected else 1.0

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": "dry_run" if args.dry_run else args.backend,
        "hardware": {
            "platform": platform.platform(),
            "cpu_threads_available": os.cpu_count(),
            "vram_used_mb": _query_vram_used_mb(),
        },
        "config": {
            "model": str(args.model) if args.model else "dry-run",
            "model_size_mb": model_size_mb,
            "n_ctx": args.n_ctx,
            "n_threads": args.n_threads,
            "n_gpu_layers": args.n_gpu_layers,
        },
        "summary": {
            "sample_count": len(metrics),
            "median_ttft_ms": round(statistics.median(m.ttft_ms for m in metrics), 2),
            "median_decode_tps": round(statistics.median(m.decode_tps for m in metrics), 2),
            "glossary_hit_rate": glossary_hit_rate,
        },
        "samples": [asdict(m) for m in metrics],
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KotoVN quantization benchmark runner")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to eval JSONL file")
    parser.add_argument("--backend", default="llama_cpp", choices=["llama_cpp"])
    parser.add_argument("--model", default=None, help="Path to local quantized GGUF model")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="0 = strict CPU inference")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate benchmark flow without model"
    )
    parser.add_argument("--output", default=None, help="Optional path to write JSON report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Benchmark report saved to {out_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
