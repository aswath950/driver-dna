PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
RUFF    := .venv/bin/ruff
MYPY    := .venv/bin/mypy
PYTEST  := .venv/bin/pytest

.PHONY: help install-dev lint type-check test ci

help:
	@echo "Available targets:"
	@echo "  install-dev   Install all dev dependencies into .venv"
	@echo "  lint          Run ruff on src/ and tests/"
	@echo "  type-check    Run mypy on race_engine.py and openf1.py"
	@echo "  test          Run pytest with coverage"
	@echo "  ci            Run lint, type-check, and test in sequence"

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

lint:
	$(RUFF) check src/ tests/

type-check:
	$(MYPY) src/race_engine.py src/openf1.py \
		--ignore-missing-imports \
		--no-error-summary

test:
	$(PYTEST) tests/ \
		--cov=src \
		--cov-report=xml \
		--cov-report=term-missing \
		-v

ci: lint type-check test
