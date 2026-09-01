# Hybrid Theory (KYT Engine)

Движок анализа криптовалютных транзакций для банковского AML-комплаенса.

Обучен на датасете **Elliptic Bitcoin** (203,769 транзакций, 46,564 размеченных). Извлекает 191 признак (165 статистических + 26 поведенческих) + графовые эмбеддинги. Классифицирует транзакции ансамблем LightGBM + K-Score + VAE + External Labels. На инфраструктуре **Iceberg + Kafka + Flink + Spark + Redis**.

Основная идея — **многоуровневый конвейер**: real-time ingestion (RPC → Kafka → Flink) → distributed feature extraction (Spark) → multi-model ensemble scoring (Unified Scorer) → triage-based case management → active learning feedback loop.

## Результаты

| Модель | Precision | Recall (illicit) | F1 | AUC-ROC | AUC-PR | Eval set |
|--------|-----------|------------------|----|---------|--------|----------|
| LightGBM | 0.977 | 0.999 | 0.988 | 0.9549 | 0.9953 | val (t=37-44) |
| Autoencoder | 0.925 | 1.000 | 0.961 | 0.5609 | 0.9331 | val (t=37-44) |
| Stacking Ensemble | 0.968 | 0.999 | 0.983 | 0.8583 | 0.9946 | test (t=45-49) |
| **Unified Scorer** | — | — | — | — | — | **production** |
| **K-Score** | — | — | — | — | — | mean=0.162, GREEN=41,434, YELLOW=4,987, RED=143 |
| **Triage** | — | — | — | — | — | **99.7% Priority, 0.3% Escalation** |

![Сравнение моделей](docs/figures/model_comparison.png)

![Матрицы ошибок](docs/figures/confusion_matrices.png)

![ROC-кривые](docs/figures/roc_curves.png)

![Важность признаков](docs/figures/feature_importance.png)

![Временное распределение](docs/figures/temporal_distribution.png)

![Анализ дрифта](docs/figures/drift_analysis.png)

## Архитектура

```mermaid
flowchart LR
    subgraph INGESTION["Data Ingestion"]
        RPC["Blockchain RPC\n(Bitcoin/Ethereum)"]
        EXT["External Feeds\n(OFAC, GoPlus, ScamDB)"]
        RPC --> KAFKA["Kafka: raw_txs"]
        EXT --> ICEBERG_RAW["Iceberg: raw_external_labels"]
    end

    subgraph STREAMING["Stream Processing (Flink)"]
        KAFKA --> FLINK["Flink SQL\nFeature Computation"]
        FLINK --> ICEBERG_FEAT["Iceberg: features"]
    end

    subgraph BATCH["Batch Processing (Spark)"]
        ICEBERG_FEAT --> STAT["StatFeatureExtractor\n(166 features)"]
        ICEBERG_FEAT --> BEHAV["BehaviorFeatureExtractor\n(26 features)"]
        ICEBERG_FEAT --> GRAPH["GraphFeatureExtractor\n(4 features)"]
        ICEBERG_FEAT --> EMB["EmbeddingGenerator\n(Node2Vec 64-d)"]
        STAT & BEHAV & GRAPH & EMB --> ICEBERG_FEAT_FULL["Iceberg: features (full)"]
    end

    subgraph TRAINING["ML Training"]
        ICEBERG_FEAT_FULL --> TRAIN["ModelTrainer\nLightGBM / VAE / Ensemble"]
        TRAIN --> MLFLOW["MLflow Tracking"]
        TRAIN --> ICEBERG_MODEL["Iceberg: models registry"]
    end

    subgraph INFERENCE["Inference API (FastAPI)"]
        API["REST /predict\n/batch-predict"]
        REDIS["Redis Cache\nFeature Store"]
        MODEL_LOADER["ModelLoader\n(MLflow + Iceberg)"]
        SCORER["UnifiedScorer\nLGBM + K-Score + VAE + External"]
        TRIAGE["TriageSystem\nauto_close / priority / escalation"]
        SHAP["SHAP Explainer"]
        
        API --> REDIS
        API --> MODEL_LOADER
        MODEL_LOADER --> SCORER
        SCORER --> TRIAGE
        SCORER --> SHAP
    end

    subgraph ACTIVE["Active Learning"]
        SCORER --> AL_SAMPLER["UncertaintySampler\nentropy + K-Score"]
        AL_SAMPLER --> ANALYST["Analyst Labeling"]
        ANALYST --> FEEDBACK["FeedbackLoop\nincremental retrain"]
        FEEDBACK --> TRAIN
    end

    subgraph MONITORING["Observability"]
        PROM["Prometheus\nMetrics"]
        BIGQUERY["BigQuery\nAnalytics"]
        GREAT_EXP["Great Expectations\nData Quality"]
        
        API --> PROM
        ICEBERG_FEAT --> GREAT_EXP
        ICEBERG_FEAT --> BIGQUERY
    end
```

## Data Lakehouse

Промышленная версия KYT Engine построена на data lakehouse-архитектуре с Apache Iceberg в качестве unified storage layer. Lakehouse сочетает гибкость data lake (произвольные форматы, schema-on-read) с транзакционными гарантиями data warehouse (ACID, time-travel, schema evolution).

**Ключевые Iceberg-таблицы:**

| Таблица | Партиция | Назначение |
|---------|----------|-----------|
| `raw_txs` | days(timestamp) | Сырые блокчейн-транзакции из Kafka + исторические CSV |
| `raw_external_labels` | days(timestamp) | OFAC/GoPlus/ScamDB лейблы с confidence scoring |
| `features` | days(timestamp) | Полный набор из 196 признаков (166 stat + 26 behavior + 4 graph + 64-d embedding) |
| `predictions` | days(timestamp) | Результаты Unified Scorer с risk_score, risk_zone, triage_level, SHAP |
| `models` | — | Версионированный реестр моделей с метриками и snapshot-ID обучающих данных |

**Преимущества lakehouse-подхода:**

- **ACID-транзакции:** атомарные коммиты предотвращают чтение частично обогащённых features
- **Time-travel:** воспроизведение предсказаний на конкретный момент времени для аудита
- **Schema evolution:** добавление колонок (stat_feat_167, embedding_65) без миграции
- **Partition evolution:** смена стратегии партиционирования без перезаписи данных
- **Hidden partitioning:** партиция по дням, прозрачная для пользователя

## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Обучение на реальных данных Elliptic
python -m kyt_engine.training.train_real

# Запуск API
make serve
# http://0.0.0.0:8000
```

## Структура

```
Hybrid-Theory/
├── configs/config.yaml
├── data/
│   ├── raw/                    # Elliptic CSV
│   └── external/               # OFAC, GoPlus, ScamDB cache
├── docs/                        
│   ├── README.md                # Detailed documentation
│   ├── methodology.md           # Technical methodology
│   ├── results.md               # Evaluation results
│   └── figures/                 # Graphs
├── models/                      # Serialized models (.pkl)
├── notebooks/                   # Jupyter (EDA, training, evaluation)
├── src/kyt_engine/
│   ├── data/                    # Loaders, scrapers, Iceberg store
│   ├── features/                # Feature engineering + Spark extractors
│   ├── ingestion/               # Kafka producer, Flink job
│   ├── metrics/                 # Metrics + SHAP
│   ├── models/                  # LightGBM, Autoencoder, K-Score, Triage, UnifiedScorer
│   ├── training/                # train_real, active_learning
│   └── api/                     # FastAPI inference service
└── tests/                       
```

## Стек

Python 3.10+ | LightGBM | PyTorch (VAE) | NetworkX + Node2Vec | FastAPI | Redis | Kafka | Flink | Spark | Iceberg | MLflow | Prometheus | Pytest

## Лицензия

GNU Affero General Public License — Copyright (c) 2026 Kirill Yuzhakov