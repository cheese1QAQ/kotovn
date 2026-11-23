UV ?= uv
BACKEND_DIR := backend

.PHONY: help sync lint format typecheck test check doctor demo bench-dry desktop-build

help:
	@echo "KotoVN Development Commands:"
	@echo "  make sync          Install backend dev dependencies via uv"
	@echo "  make lint          Run ruff lint & format check (backend + research)"
	@echo "  make format        Auto-format Python code with ruff"
	@echo "  make typecheck     Run strict mypy type checking"
	@echo "  make test          Run pytest suite with coverage"
	@echo "  make check         Run full backend CI gate (lint + typecheck + test)"
	@echo "  make doctor        Inspect local CPU/VRAM and recommended profile"
	@echo "  make demo          Run offline mock VN translation + online RAG demo"
	@echo "  make bench-dry     Validate quantization benchmark dataset & runner"
	@echo "  make desktop-build Typecheck and build desktop frontend"

sync:
	cd $(BACKEND_DIR) && $(UV) sync --extra dev

lint:
	cd $(BACKEND_DIR) && $(UV) run ruff check src tests ../research
	cd $(BACKEND_DIR) && $(UV) run ruff format --check src tests ../research

format:
	cd $(BACKEND_DIR) && $(UV) run ruff check --fix src tests ../research
	cd $(BACKEND_DIR) && $(UV) run ruff format src tests ../research

typecheck:
	cd $(BACKEND_DIR) && $(UV) run mypy

test:
	cd $(BACKEND_DIR) && $(UV) run pytest --cov=kotovn --cov-report=term-missing

check: lint typecheck test

doctor:
	cd $(BACKEND_DIR) && $(UV) run kotovn doctor

demo:
	cd $(BACKEND_DIR) && $(UV) run kotovn demo

bench-dry:
	cd $(BACKEND_DIR) && $(UV) run python ../research/quantization/benchmark_runner.py --dry-run

desktop-build:
	cd desktop && npm run build
