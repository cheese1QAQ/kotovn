# %% [markdown]
# # KotoVN 离线 LLM 轻量化实验：GGUF + VN 领域 `imatrix` 校准量化 (Google Colab)
#
# > **用途说明**：本脚本专门运行在 **Google Colab GPU** 环境，用于离线模型压缩与量化消融研究，
# > **不用于部署任何在线服务**。实验完成后，请将筛选出的最优 `.gguf` 文件下载至本地运行。
#
# ## 实验目标
# 1. 将 FP16/BF16 Hugging Face 模型转换为 `GGUF (F16)` 中间权重。
# 2. 基于 Visual Novel 双语对话语料生成领域专属 **Importance Matrix (`imatrix`)**。
# 3. 批量量化 `Q8_0`、`Q4_K_M`、`IQ4_XS`、`IQ3_M` 四个档位，并对比体积与困惑度（PPL）。
# 4. 导出最优轻量化权重回 Google Drive / 本地，再交由 `benchmark_runner.py`
#    在本地做纯 CPU 与 12GB 显存混合实测。

# %%
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantExperimentConfig:
    """Editable parameters for a single Colab quantization sweep."""

    hf_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    work_dir: Path = Path("/content/kotovn_quant_lab")
    llama_cpp_dir: Path = Path("/content/llama.cpp")
    quant_types: tuple[str, ...] = ("Q8_0", "Q4_K_M", "IQ4_XS", "IQ3_M")
    imatrix_chunks: int = 80
    ppl_ctx_size: int = 512


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def build_vn_calibration_corpus(eval_jsonl: Path, output_txt: Path) -> Path:
    """Build a VN-domain bilingual calibration corpus for llama-imatrix."""
    lines: list[str] = []
    if eval_jsonl.exists():
        for raw in eval_jsonl.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            item = json.loads(raw)
            glossary_str = ", ".join(f"{k}={v}" for k, v in item["glossary"].items())
            block = (
                f"[SPEAKER:{item['speaker']}|STYLE:{item['archetype']}|GLOSSARY:{glossary_str}]\n"
                f"JA: {item['source']}\n"
                f"ZH: {item['reference']}\n"
            )
            lines.append(block)
    if not lines:
        raise RuntimeError(f"No calibration entries found in {eval_jsonl}")

    # Repeat small starter corpus for smoke runs; swap in a full VN corpus for real sweeps.
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(lines * 20), encoding="utf-8")
    print(f"Wrote VN calibration corpus: {output_txt} ({output_txt.stat().st_size} bytes)")
    return output_txt


def run_colab_quantization_pipeline(cfg: QuantExperimentConfig, eval_jsonl: Path) -> None:
    """End-to-end offline quantization workflow intended for Google Colab GPU."""
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    model_dir = cfg.work_dir / "hf_model"
    f16_gguf = cfg.work_dir / "model-f16.gguf"
    calib_txt = cfg.work_dir / "vn_calib.txt"
    imatrix_dat = cfg.work_dir / "vn_imatrix.dat"

    # Step 1: Build llama.cpp with CUDA acceleration in Colab.
    if not cfg.llama_cpp_dir.exists():
        run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggml-org/llama.cpp",
                str(cfg.llama_cpp_dir),
            ]
        )
    run_cmd(
        ["cmake", "-B", "build", "-DGGML_CUDA=ON"],
        cwd=cfg.llama_cpp_dir,
    )
    run_cmd(
        [
            "cmake",
            "--build",
            "build",
            "--config",
            "Release",
            "-j",
            "--target",
            "llama-quantize",
            "llama-imatrix",
            "llama-perplexity",
        ],
        cwd=cfg.llama_cpp_dir,
    )

    # Step 2: Download FP16/BF16 weights and convert to F16 GGUF.
    run_cmd(["huggingface-cli", "download", cfg.hf_model_id, "--local-dir", str(model_dir)])
    run_cmd(
        [
            "python3",
            str(cfg.llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(model_dir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ]
    )

    # Step 3: Compute VN-domain Importance Matrix (imatrix).
    build_vn_calibration_corpus(eval_jsonl, calib_txt)
    run_cmd(
        [
            str(cfg.llama_cpp_dir / "build/bin/llama-imatrix"),
            "-m",
            str(f16_gguf),
            "-f",
            str(calib_txt),
            "-o",
            str(imatrix_dat),
            "--chunks",
            str(cfg.imatrix_chunks),
            "-ngl",
            "99",
        ]
    )

    # Step 4: Quantize across target bit-widths and measure perplexity.
    for qtype in cfg.quant_types:
        out_gguf = cfg.work_dir / f"model-{qtype}-vn-imatrix.gguf"
        run_cmd(
            [
                str(cfg.llama_cpp_dir / "build/bin/llama-quantize"),
                "--imatrix",
                str(imatrix_dat),
                str(f16_gguf),
                str(out_gguf),
                qtype,
            ]
        )
        size_mb = out_gguf.stat().st_size / (1024 * 1024)
        print(f"[DONE] {qtype}: {out_gguf.name} ({size_mb:.1f} MiB)")

        run_cmd(
            [
                str(cfg.llama_cpp_dir / "build/bin/llama-perplexity"),
                "-m",
                str(out_gguf),
                "-f",
                str(calib_txt),
                "-c",
                str(cfg.ppl_ctx_size),
                "--chunks",
                "16",
                "-ngl",
                "99",
            ]
        )


if __name__ == "__main__":
    print(
        "This script is designed for offline quantization experiments in Google Colab.\n"
        "Inspect QuantExperimentConfig and invoke run_colab_quantization_pipeline() inside Colab."
    )
