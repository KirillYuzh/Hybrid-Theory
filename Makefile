.PHONY: install test lint clean generate validate validate-real

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

generate:
	$(PYTHON) -m kyt_engine.synth generate --config configs/generator_behavior.yaml

validate:
	$(PYTHON) -m kyt_engine.synth validate --dir data/synthetic/behavior_run

validate-real:
	$(PYTHON) -m kyt_engine.synth validate-real --config configs/generator_behavior.yaml --raw-dir data/raw --out data/synthetic/behavior_run/strict_report.json
