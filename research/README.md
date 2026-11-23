# KotoVN 离线研究实验室（LLM 轻量化与轻量视觉）

> **边界约定**：
> - **`research/` 与 Google Colab**：仅用于离线模型压缩、量化校准、视觉模块消融与评测实验。
> - **`backend/` 与 `desktop/`**：100% 本地运行时，不依赖任何云端或 Colab 推理服务。

## 目录结构

* [`quantization/`](./quantization/)：LLM 量化与轻量化主研究方向
  - `eval_dataset.jsonl`：视觉小说专项评测集（覆盖专有名词、人称省略、敬语、傲娇/大小姐/三无口癖）
  - `benchmark_runner.py`：统一基准测评工具（支持 `--dry-run` 校验与本地 `llama.cpp` 纯 CPU / 12GB 混合显存实测）
  - `colab_quantize_gguf_imatrix.py`：在 Google Colab 上执行 FP16 -> GGUF + VN 领域 `imatrix` 校准量化的可复现脚本
* [`vision/`](./vision/)：轻量视觉模块研究（ONNX OCR + 帧差按需视觉标签 vs 紧凑型 VLM）
