ifneq ($(wildcard /sgoinfre),)
SGOINFRE_USER := $(shell whoami)
export HF_HOME := /sgoinfre/$(SGOINFRE_USER)/hf_cache
export UV_CACHE_DIR := /sgoinfre/$(SGOINFRE_USER)/uv_cache
endif
PYTHON=.venv/bin/python3


install:
	uv sync

run: install
	uv run $(PYTHON) -m src

debug:
	$(PYTHON) -m pdb $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .venv

lint:
	uv run flake8 . --exclude=llm_sdk,.venv
	uv run mypy . --exclude '\.venv' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=llm_sdk,.venv
	uv run mypy . --exclude '\.venv' --strict

.PHONY: install run debug clean lint lint-strict
