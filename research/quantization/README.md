# LLM 轻量化与量化实验指南

完整研究设计与课题说明请参阅 [docs/quantization-research.md](../../docs/quantization-research.md)。

## 快速验证测评脚本（无需模型权重，CI 同款）

```bash
python3 research/quantization/benchmark_runner.py --dry-run
```

## 在 Google Colab 上运行离线量化实验

1. 打开 Google Colab，选择带 GPU 的运行时（T4 16GB / L4 24GB / A100）。
2. 将 `colab_quantize_gguf_imatrix.py` 导入为 Notebook（或在 Colab 中克隆本仓库后按块执行）。
3. 实验产出：
   - 通用 WikiText `imatrix` vs VN 领域 `imatrix` 的量化对比权重（`Q8_0`, `Q4_K_M`, `IQ4_XS`, `IQ3_M`）
   - 各量化版本的 PPL 与体积报告
4. 将最终胜出的 `.gguf` 权重下载回本机，用本地 `benchmark_runner.py` 实测纯 CPU（`--n-gpu-layers 0`）与 12GB 混合卸载（`--n-gpu-layers 24`）下的真实 `TTFT` 和 `tok/s`。

## 本地实测命令示例（安装 `llama-cpp-python` 后）

```bash
# 纯 CPU 基准测试（目标：零显存占用）
python3 research/quantization/benchmark_runner.py \
  --backend llama_cpp \
  --model /path/to/model-IQ4_XS-vn-imatrix.gguf \
  --n-gpu-layers 0 \
  --n-threads 8 \
  --output research/quantization/reports/cpu_iq4_xs.json

# 12GB 显存混合卸载对比测试
python3 research/quantization/benchmark_runner.py \
  --backend llama_cpp \
  --model /path/to/model-Q4_K_M-vn-imatrix.gguf \
  --n-gpu-layers 24 \
  --n-threads 8 \
  --output research/quantization/reports/hybrid12gb_q4_k_m.json
```
