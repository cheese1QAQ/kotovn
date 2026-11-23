# 贡献与协作指南（GitHub 仓库维护规范）

本项目严格遵循现代开源工程实践，同时作为系统学习 GitHub 协作与 CI/CD 的实战载体。

## 1. 分支模型（GitHub Flow）

- `main` 为受保护主干分支，始终保持可构建、CI 全绿状态。
- 所有开发均从 `main` 切出短生命周期分支，并通过 Pull Request 合并：
  - `feat/<topic>`：新功能（例如 `feat/llama-cpp-backend`）
  - `fix/<topic>`：缺陷修复（例如 `fix/glossary-longest-match`）
  - `research/<topic>`：轻量化/视觉研究实验与报告（例如 `research/vn-imatrix-iq4xs`）
  - `docs/<topic>`：文档更新
  - `chore/<topic>`：工程配置、依赖与 CI 维护

## 2. 提交规范（Conventional Commits）

采用 `<type>(<scope>): <subject>` 格式，便于自动生成 Changelog 与 Release Notes：

- `feat(rag): add sqlite-vec persistent hybrid store`
- `perf(quant): benchmark vn-domain imatrix on IQ4_XS`
- `fix(desktop): prevent overlay from stealing game focus`
- `ci(actions): add python 3.12 matrix and coverage report`
- `docs(arch): document hybrid 12GB offload thresholds`

## 3. 提交 PR 前本地检查清单

```bash
make check       # ruff lint + format check + mypy --strict + pytest
make bench-dry   # 校验 research/quantization 数据集与脚本
cd desktop && npm run build
```

## 4. 轻量化研究实验提交规范

提交 `research/` 下的实验结论时，请务必保留：
1. 实验脚本或确切命令行参数（确保可复现）
2. 本地硬件环境与配置（CPU 型号、线程数、GPU 显存、`n_gpu_layers`、KV Cache 精度）
3. 由 `benchmark_runner.py` 导出的结构化 JSON 报告（不要将 GB 级 `.gguf` 模型权重提交到 Git！）
