# 04-structure.md — Структура после перестройки

```
Hybrid-Theory/
├── README.md                      # переписан: генератор для Spillety, как запускать
├── pyproject.toml                 # переписан: deps numpy/pandas/scipy/pyyaml, tooling (ruff, pytest, mypy)
├── Makefile                       # переписан: install/test/lint/stats/generate/validate
├── configs/
│   └── generator.yaml             # конфиг генератора (см. 03-design)
├── data/
│   ├── raw/                       # НЕ ТРОГАЕМ: elliptic_* (689МБ) — gitignored, источник статистик
│   └── elliptic_stats/            # артефакты статистик (эмпирические CDF), создаются скриптом
├── src/kyt_engine/
│   ├── __init__.py
│   ├── _stats/
│   │   └── compute.py             # разовый скрипт: Elliptic → CDF-артефакты + counts/step
│   └── synth/
│       ├── __init__.py
│       ├── config.py              # dataclass GeneratorConfig (загрузка из yaml)
│       ├── stats.py               # загрузка CDF-артефактов, inverse-CDF + jitter
│       ├── schemes.py             # шаблоны схем (mixer/peel/fanout/hub/wash/p2p)
│       ├── graph.py               # сборка графа: роли, рёбра, time_step (+дрифт)
│       ├── features.py            # генерация feat_2..feat_166 по ролям
│       ├── emit.py                # запись 3 CSV + manifest.json (sha256)
│       ├── validate.py            # дубль контракта spillety/data/loader.py
│       └── __main__.py            # CLI: generate | stats | validate
├── tests/
│   ├── test_contract.py           # AC-1,2: формат тождествен loader.py, детерминизм
│   ├── test_schemes.py            # AC-3: схемы из manifest восстановимы, классы согласованы
│   └── test_drift.py              # AC-4: shutdown_step снижает illicit-долю в поздних шагах
└── TASKS/pivot-to-synthetic-data/ # артефакты QRSPI (текущая задача)
```

## Удаляется напрямую (старое содержимое)

- `models/`, `docs/` (фейковые результаты/фигуры), `notebooks/`, `demo/`, `reports/`, `infra/`, `ansible/`, `k8s/`, `deploy.sh`, `docker-compose.yml`, `Dockerfile`, `test_predict_local.py`, `test_small.py`, `temp.md`, `.pre-commit-config.yaml`
- `src/kyt_engine/{api,core,dashboard,data,features,filtering,graph_metrics,metrics,models,security,simulations,training}/`
- `configs/config.yaml` (заменяется)

## Файлы пользователя, которые НЕ трогаем

- `data/raw/*` (в .gitignore), `K-BRAIN/`, `.venv/`, `data/external/*`