# 轻量高效 LLM 视觉模块研究规划（目标 #2）

> 本目录用于对比视觉小说读屏场景下的轻量 OCR 与紧凑型 VLM 方案，核心衡量标准：**CPU 友好度、显存开销、视觉 Token 数量与表情/场景语义增益**。

## 计划对比的三条路线

1. **路线 A（CPU-First 纯轻量 OCR）**：
   - `RapidOCR` / `PaddleOCR-ONNX` / `Manga-OCR-ONNX`
   - 配合 **StabilityGate（帧差防抖）** 与 **Auto-ROI（对话框自动定位）**，零显存占用。
2. **路线 B（两阶段按需视觉增强 — 推荐 12GB 混合档）**：
   - 文本固定由轻量 OCR 提取保证零幻觉；
   - 通过立绘/背景区域的感知哈希（pHash/SSIM）监测画面突变，仅在立绘表情或场景切换时触发轻量视觉模型产出 `{"scene_tag", "speaker_emotion"}`。
3. **路线 C（端到端紧凑 VLM）**：
   - 实验 1B–4B 紧凑多模态架构（`Qwen2.5-VL-3B`、`MiniCPM-V`、`SmolVLM` 等 GGUF/量化版）。
   - 重点研究：**ROI 裁剪 + 最大动态分辨率切片控制**如何将单帧视觉 Token 从 1000+ 压缩到 ~128–256 Tokens，以换取可接受的本地 TTFT。
