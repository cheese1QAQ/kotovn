# LLM 轻量化与量化研究方案（研究主线 #1）

> **核心定位**：Google Colab 算力**专门用于离线 LLM 轻量化研究**（量化转换、校准矩阵生成、消融评测与方法对比），**绝不用于在线翻译服务部署**。研究产出的最优轻量化模型权重导出回本地，在 **CPU（Ryzen 7 5800X3D）** 与 **12GB 显存（RTX 5070）** 环境下完成最终实测与运行。

---

## 1. 研究问题定义

视觉小说翻译场景对本地 LLM 提出了非常具体的约束：
1. **显存受限且希望尽量 CPU 可跑**：本机只有 12GB 显存且可能同时运行游戏，理想目标是将翻译模型压缩到 **3GB–6GB** 内存/显存占用。
2. **对长程指令遵循和风格极其敏感**：粗暴的低比特量化（如无校准 3-bit/4-bit）容易导致**人物口癖退化、人称代词错乱、JSON 结构化输出格式崩溃、或者中英日混杂**。
3. **低延迟与长上下文的矛盾**：VN 需要维护几十轮对话 + RAG 术语 + 人格 Few-Shot（约 2K–4K tokens 上下文），**KV Cache 体积与 Prefill 速度**往往比权重本身更早成为瓶颈。

因此，本研究围绕四个递进课题展开：

---

## 2. 四大研究课题与实验矩阵

### 课题 A：PTQ 训练后量化方法横向对比（权重压缩）
在 Colab（T4/L4/A100）上选取同一基座模型（建议先选 **7B/8B** 如 `Qwen2.5-7B-Instruct` 或日译中微调模型，外加 **3B/4B** 对照组测试纯 CPU 极限），对比不同量化格式与算法：

| 方法类别 | 具体配置 | 核心原理学习点 | 适用本地后端 |
| :--- | :--- | :--- | :--- |
| **Baseline** | `FP16` / `BF16` | 全精度基准（Colab 测质量上限） | PyTorch / vLLM |
| **均匀对称/非对称量化** | `GGUF Q8_0`, `Q4_0` | 分组缩放因子（Group-wise Quantization）与反量化开销 | `llama.cpp` (CPU/GPU) |
| **混合精度 K-Quants** | `GGUF Q4_K_M`, `Q5_K_M`, `Q3_K_M` | 对 Attention `v`/`o` 与 FFN `down` 等敏感层分配更高比特 | `llama.cpp` (CPU/GPU) |
| **重要性矩阵 I-Quants** | `GGUF IQ4_XS`, `IQ3_M` + **`imatrix`** | 基于校准集激活分布估算权重 Hessian/重要性，非均匀查表量化 | `llama.cpp` (CPU/GPU) |
| **激活感知量化（对比研究）** | `AWQ (W4A16)`, `GPTQ` | 保护激活幅值最高的 1% 显著通道（Salient Channels） | AutoAWQ / ExLlamaV2 (GPU) |
| **硬件前沿探索（可选）** | `FP8` / `NVFP4`（利用 RTX 5070 Blackwell 特性） | 新一代低精度浮点格式与动态缩放对比传统 INT4 | TensorRT-LLM / llama.cpp |

### 课题 B：领域校准集（`imatrix`）对 VN 翻译质量的提升（核心亮点实验）
常规公开量化模型通常使用英文 WikiText 作为 `imatrix` 校准集，在**日译中二次元口语 + 敬语 + 专有名词**场景下并非最优。
* **实验假设**：使用混合「日文 VN 对话 + 高质量中译 + 结构化术语提示词」构建专属校准语料（`vn_calib.txt`）计算 `imatrix`，在同等 `IQ4_XS` / `Q4_K_M` 体积下，能够显著降低口癖丢失与格式错误率。
* **对照实验**：
  1. 无 `imatrix` 直接量化 `Q4_K_M` / `IQ4_XS`
  2. 通用英文 WikiText `imatrix` 量化
  3. **VN 领域双语 `imatrix` 量化**

### 课题 C：KV Cache 量化与上下文压缩（面向长对话与纯 CPU 推理）
* 对比 `KV Cache FP16` vs `Q8_0` vs `Q4_0` 在 4096 上下文长度下的：
  - 内存/显存节省比例
  - 长距离人物称呼一致性（隔 20 轮对话是否仍记得术语与关系）
  - CPU 内存带宽瓶颈下的解码速度（tok/s）变化

### 课题 D：CPU 极致轻量化探索（3B–4B 小模型 + 知识蒸馏 / QLoRA 微调，进阶）
* 面向“纯 CPU 秒级出译文”的终极目标：研究 **3B/4B 模型（如 Qwen2.5-3B / Llama-3.2-3B）** 通过 Colab 上 **QLoRA（NF4 + LoRA）** 吸收 7B/14B 教师模型的 VN 翻译与人格模仿能力（任务蒸馏），再合并权重并导出 `GGUF Q4_K_M`（仅 ~2GB），在 5800X3D 纯 CPU 上挑战 20+ tok/s。

---

## 3. 统一评测指标体系

所有实验结果通过 `research/quantization/benchmark_runner.py` 统一记录为 JSON 报告，便于追踪与绘图：

1. **效率与资源指标（本地实测）**：
   - `model_size_mb`：磁盘体积
   - `peak_ram_mb` / `peak_vram_mb`：峰值内存与显存
   - `ttft_ms`：首字延迟（Prefill 耗时，直接影响玩家阅读体感）
   - `decode_tps`：生成吞吐量（tokens/s）
   - `cpu_threads` / `n_gpu_layers`：记录确切硬件与卸载配置
2. **质量与退化指标（Colab / 本地评测集 `eval_dataset.jsonl`）**：
   - **困惑度（PPL）**：在日文/双语 VN 验证集上的 PPL 相对 FP16 的增幅（$\Delta\text{PPL}$）
   - **术语遵循率（Glossary Hit Rate）**：Prompt 给出 RAG 术语后，译文精确命中的比例（0–100%）
   - **结构/格式合法率**：是否严格按约定格式输出，不出现多余解释或乱码
   - **人格一致性得分（Persona Fidelity）**：结合规则检测（第一人称/句尾口癖）+ 强模型 LLM-as-a-Judge 盲评打分（1–5 分）

---

## 4. Colab 与本地的协作工作流

```text
[本地仓库 research/quantization/]
  1. 维护 eval_dataset.jsonl 与校准语料生成脚本
  2. 推送至 GitHub 或挂载 Google Drive
           │
           ▼
[Google Colab GPU (仅用于离线研究)]
  3. 运行 colab_quantize_gguf_imatrix.py：
     - 拉取 FP16 基座模型
     - 编译/调用 llama.cpp
     - 生成通用 vs VN 领域 imatrix
     - 批量量化 (Q8_0 / Q4_K_M / IQ4_XS / IQ3_M)
     - 跑 PPL 与质量自动打分，保存实验 JSON 报告
           │
           ▼
[下载最优 GGUF 权重回到本地]
  4. 本地执行 benchmark_runner.py：
     - 实测纯 CPU (n_gpu_layers=0) 与混合 12GB (n_gpu_layers=N) 的 TTFT & tok/s
     - 将实验结论提交到 research/quantization/reports/ 并在 README 更新对比表
```
