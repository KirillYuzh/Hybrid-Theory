FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e ".[dev]" 2>/dev/null || pip install pandas numpy lightgbm torch scikit-learn fastapi uvicorn plotly dash pyiceberg duckdb

COPY src/ src/
COPY configs/ configs/
COPY models/ models/

EXPOSE 8000 8050
CMD ["python", "-m", "uvicorn", "kyt_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
