.PHONY: install test lint clean stats generate validate

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache .pytest_cache

stats:
	$(PYTHON) -m kyt_engine._stats.compute

generate:
	$(PYTHON) -m kyt_engine.synth generate --config configs/generator.yaml

validate:
	$(PYTHON) -m kyt_engine.synth validate --dir data/synthetic/run