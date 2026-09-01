install:
    pip install -e ".[dev]"

test:
    python -m pytest tests/ -v

train:
    python -m kyt_engine.training

serve:
    uvicorn kyt_engine.api.app:app --reload --host 0.0.0.0 --port 8000

lint:
    ruff check src/ tests/
    ruff format src/ tests/

clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
