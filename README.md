# KotoVN — 本地优先、用户无感的视觉小说 LLM + 动态 RAG 翻译器

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](backend/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**KotoVN** 是一款基于 **本地轻量化 LLM + 轻量视觉模块（OCR/VLM）+ 在线维护 RAG（边翻边学）+ 人物人格模板** 的视觉小说翻译器与开源工程研究项目。

---

## 核心特色与设计边界

1. **100% 本地推理，CPU 优先 & 12GB 显存友好**
   - 默认提供 **纯 CPU 档位（VRAM = 0 GB）** 与 **12GB 混合卸载档位**，基于 `llama.cpp` (GGUF) + `ONNX Runtime`，不与游戏争抢显存。
   - **Google Colab 算力严格仅用于离线 LLM 轻量化研究**（`research/quantization/`），绝不用于在线翻译服务部署。
2. **用户无感交互（Zero-Friction UX）**
   - 仅需初始两步引导（模型档位 + 游戏窗口），即可进入自动跟随窗口、透明、鼠标穿透的悬浮字幕状态。
   - 输入优先基于 **OCR/VLM 实时读屏**（开箱即用，无需折腾解包），同时在核心接口层为未来的剧本解包器预留了 `ScriptSource` 扩展点。
3. **混合 RAG 与“边翻边学”**
   - 三层架构（精确术语表 + 语义记忆库 + 人物关系图），后台异步 `Reflector` 边翻边学新词与剧情摘要，并支持 **Save/Load 读档 Checkpoint 快照回滚** 防止多路线串味。
4. **鲜活的人物人格模板**
   - 内置跨作品角色原型范本库（傲娇、大小姐、三无等），支持根据角色台词动态覆盖语言学指纹与 Few-Shot 风格锚点。

---

## 项目目录结构

```text
kotovn/
├── .github/                  # GitHub Actions CI/CD、Dependabot、Issue & PR 模板
├── docs/                     # 详细架构设计、LLM 轻量化研究方案、UX 规范与路线图
├── backend/                  # Python 核心引擎 (kotovn): 接口协议、RAG、人格引擎、管线、单测
├── desktop/                  # 桌面端无感 GUI 骨架 (React + TypeScript，预备接入 Tauri v2)
├── research/                 # 离线研究实验室 (Google Colab 量化脚本 + 统一 Benchmark 套件)
├── Makefile                  # 统一开发命令入口
└── CONTRIBUTING.md           # GitHub Flow、Conventional Commits 与实验记录规范
```

---

## 快速开始（Phase 0 验证）

```bash
# 1. 安装后端开发环境并运行完整质量门禁（无需 GPU）
make sync
make check

# 2. 运行硬件诊断与离线 Mock 翻译 + 边翻边学演示
make doctor
make demo

# 3. 校验轻量化研究评测集与 Benchmark 管线（干跑模式）
make bench-dry

# 4. 启动或构建前端无感界面预览
cd desktop && npm install && npm run build
```

---

## 文档导航

* [系统总体架构设计](docs/architecture.md)
* [LLM 轻量化与量化研究方案（Colab 离线实验 -> 本地部署）](docs/quantization-research.md)
* [动态 RAG（边翻边学）与人物人格模板设计](docs/rag-and-persona.md)
* [用户无感（Zero-Friction）图形界面交互规范](docs/ux-zero-friction.md)
* [分阶段里程碑路线图（Phase 0 → Phase 4）](docs/roadmap.md)
