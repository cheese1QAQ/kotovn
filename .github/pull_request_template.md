## 变更摘要

<!-- 简述本次 PR 解决的问题或新增的能力 -->

## 关联阶段 / 议题

- Phase: <!-- Phase 0 / 1 / 2 / 3 / 4 -->
- Closes #

## 变更类型

- [ ] 新功能 (`feat`)
- [ ] 缺陷修复 (`fix`)
- [ ] LLM 轻量化 / 视觉研究实验 (`research` / `perf`)
- [ ] 文档更新 (`docs`)
- [ ] 工程 / CI 维护 (`chore` / `ci`)

## 验证检查清单

- [ ] 本地运行 `make check` 通过（`ruff` + `mypy` + `pytest`）
- [ ] 未提交模型权重文件（`.gguf` / `.safetensors` / `.onnx`）或本地游戏缓存
- [ ] 如涉及接口契约变更，已同步更新 `backend/src/kotovn/schemas/` 与 `desktop/src/types/protocol.ts`
