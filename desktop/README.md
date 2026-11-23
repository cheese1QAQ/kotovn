# KotoVN Desktop Client（Tauri v2 + React + TypeScript）

前端遵循「**用户无感（Zero-Friction）**」设计（详见 [docs/ux-zero-friction.md](../docs/ux-zero-friction.md)）：
- **初次启动**：仅需两步（选硬件档位/模型 + 选游戏窗口）。
- **日常游玩**：透明、置顶、鼠标穿透的 Overlay HUD 自动吸附游戏窗口；仅在悬停术语或按热键时展开微调。
- **进阶调试**：按需呼出的 Studio 抽屉（RAG 术语审校、Checkpoint 回滚、人格工坊、量化延迟监控）。

## 本地开发

```bash
cd desktop
npm install
npm run typecheck
npm run dev
```

> Phase 1 将执行 `npx @tauri-apps/cli init` 接入 Rust 原生窗口跟踪、全局热键与鼠标穿透能力，并将本地 Python 引擎作为 Sidecar 启动。
