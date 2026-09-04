FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000 8050
CMD ["python", "-m", "uvicorn", "kyt_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]