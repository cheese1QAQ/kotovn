# 项目分阶段迭代路线图（Roadmap）

本项目围绕 6 大学习目标划分为 5 个可独立验收的阶段，每个阶段都能产出一个可运行、可测试、可展示的里程碑版本。

---

## Phase 0：工程底座、架构文档与 CI 流水线（当前已完成初始化）
* **目标**：建立工业级 GitHub 仓库规范、核心接口抽象、零 GPU 可测环境与离线轻量化研究底座。
* **核心交付物**：
  - [x] 完整架构设计文档（系统架构、LLM 轻量化研究方案、动态 RAG & 人格设计、无感 UX 规范）
  - [x] `backend/` Python 核心包骨架（`kotovn`）：严格类型契约、协议接口、Mock 后端、内存版混合 RAG & 人格模板加载器、CLI 诊断入口
  - [x] `research/quantization/` 离线轻量化实验套件：VN 专项评测集、基准测评 runner、Colab `imatrix` GGUF 量化实验脚本
  - [x] `desktop/` 无感前端骨架（React + TypeScript + 协议类型定义）
  - [x] `.github/` CI/CD 流水线（Lint / Typecheck / Unit Test / Frontend Check / Research Check / Tag Release 模板）与社区协作模板
* **对应学习目标**：#5（完整工程化与 CI/CD）

---

## Phase 1：本地最小可行闭环（MVP） & 纯 CPU / 混合推理打通
* **目标**：在本地真正跑通“截取游戏画面 -> OCR 提取文本 -> 本地量化 LLM 流式翻译 -> 悬浮窗显示”。
* **核心任务**：
  - [ ] 接入 `llama-cpp-python`（或 `llama-server` 子进程）实现真实 `LlamaCppBackend`，支持 `cpu_only` 与 `hybrid_12gb` 切换
  - [ ] 实现跨平台窗口捕获与 `StabilityGate`（帧差防抖）+ 轻量 ONNX OCR（`RapidOCR` / `Manga-OCR`）
  - [ ] 实现 FastAPI + WebSocket 本地服务，向 `desktop/` 推送流式译文
  - [ ] 完成 Tauri v2 壳层接入：两步引导页 + 自动跟随游戏窗口的透明鼠标穿透悬浮窗
* **验收标准**：在 12GB 显卡环境下（纯 CPU 档位 VRAM = 0 GB）对任意 VN 窗口完成端到端实时翻译。
* **对应学习目标**：#1、#2、#6

---

## Phase 2：LLM 轻量化深度研究 & 轻量视觉模块实验
* **目标**：把 LLM 量化与轻量视觉模块做成可复现、有数据支撑的系统研究。
* **核心任务**：
  - [ ] 在 Google Colab 运行 `colab_quantize_gguf_imatrix.py`，对比通用 `imatrix` 与 VN 领域 `imatrix` 在 `Q8_0 / Q4_K_M / IQ4_XS / IQ3_M` 上的 PPL 与翻译质量退化曲线
  - [ ] 在本地 5800X3D + RTX 5070 上跑完 `benchmark_runner.py`，产出 CPU vs 混合卸载 vs KV Cache 量化的完整性能报告
  - [ ] 实现 **Auto-ROI 对话框自动检测**，并对比两阶段（ONNX OCR + 场景变更触发视觉标签）与端到端紧凑 VLM（2B–4B）的延迟与表情识别收益
* **验收标准**：`research/` 下产出可复现实验报告图表，形成一套兼顾纯 CPU 速度与翻译质量的推荐量化配置。
* **对应学习目标**：#1、#2

---

## Phase 3：持久化混合 RAG（边翻边学）与动态人格工坊
* **目标**：让翻译器具备“越翻越准、角色说话像活人、不怕读档串线”的核心竞争力。
* **核心任务**：
  - [ ] 将内存版 RAG 升级为持久化 **SQLite + SQLite-vec（或 LanceDB）+ ONNX Embedding + BM25** 混合检索存储
  - [ ] 上线后台异步 `Reflector` 管线：自动提取新术语（`draft` 态）、人物称呼变更与场景剧情摘要，支持冲突去重与 Checkpoint 回滚
  - [ ] 扩充跨作品范本库至 10+ 常见 VN 角色原型，实现冷启动自动匹配与在线风格微调
  - [ ] 在悬浮窗支持术语悬停原位修正（即时升级为 `verified`），在 Studio 抽屉上线人格试译对比
* **验收标准**：连续翻译 100 句 VN 对话，专有名词一致性 >= 95%，角色自称与口癖显著区分且支持读档回滚。
* **对应学习目标**：#3、#4、#6

---

## Phase 4：剧本解包扩展、交互打磨与全自动 CD 发布
* **目标**：补全“剧本驱动人格预构建”闭环，并达到开源项目可对外发布的工程水准。
* **核心任务**：
  - [ ] 基于预留的 `ScriptSource` 接口，实现主流 VN 引擎剧本导入器（优先支持通用 JSON/CSV 及 Ren'Py）
  - [ ] 实现全量脚本说话人台词聚类 -> 范本匹配与差异覆写 -> 一键生成全员人格卡片与初始术语表
  - [ ] 完善 `release.yml` 跨平台打包流水线（Windows/Linux 安装包）与完整用户使用文档
* **对应学习目标**：#4、#5、#6
