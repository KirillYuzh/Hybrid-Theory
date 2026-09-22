# Hybrid-Theory

Генератор синтетических криптовалютных транзакций в формате **Elliptic++** — источник данных и контролируемых ground-truth бенчмарков для KYT/AML-проектов (в первую очередь [Spillety](https://github.com/Spillety/Spillety)).

## Что генерирует

| Файл | Формат |
|------|--------|
| `elliptic_txs_features.csv` | без header: `txId, time_step, feat_2..feat_166` (167 колонок) |
| `elliptic_txs_classes.csv` | `txId, class`; class ∈ {"1"="illicit", "2"="licit", "unknown"} |
| `elliptic_txs_edgelist.csv` | `txId1, txId2` (рёбра) |

Плюс `manifest.json` — конфиг, seed, sha256 трёх файлов и **ground truth**: для каждой транзакции `{txId, scheme, role, class}`.

Эталонный прогон (`configs/generator.yaml`, seed 42): **10 000 tx, 14 092 рёбра**; real `spillety.data.loader.load_elliptic` читает без ошибок, `temporal_split` даёт **5856 / 2050 / 2094** (train ≤30 / valid 31–40 / test 41–49). Классы — строки `{'unknown', '2', '1'}`, как в реальном Elliptic.

## Откуда реализм

Артефакты `data/elliptic_stats/` — эмпирические CDF по каждой из 165 фич (feat_2..feat_166) для трёх классов, снятые с реального Elliptic (203 769 tx) разовым скриптом `kyt_engine._stats.compute`. Генератор делает inverse-CDF-сэмплинг по классу транзакции.

## Схемы (паттерны)

Посаженные подграфы с известной семантикой — их Spillety использует как ground-truth для retrieval/contrastive/дрифт-тестов:

- `mixer` — fan-in → mixer_core → fan-out (illicit core);
- `peel_chain` — линейная цепь, все средние узлы illicit;
- `fanout` — scam → жертвы (illicit scam);
- `hub_spoke` — обменный/рыночный хаб (licit);
- `wash` — цикл wash-trading (illicit), **дрифт-эксклюзивная схема**: генерируется только в дрифт-режиме как novel-scheme и только на поздних шагах;
- `p2p` — фон licit/unknown (и остаток illicit-бюджета): случайные направленные рёбра между фоновыми tx, **без гарантии связности** (нужны для degree/статистик, а не для связности компонент).

## Пропорции (бюджеты)

`labeled_ratio` и `illicit_ratio_in_labeled` — бюджеты: фоновые транзакции добирают их до остатка, но схемы могут дать меньше размеченных, чем бюджет (жёсткое равенство не гарантируется). Гарантируется ровно n_txs транзакций.

## Temporal drift (опция)

`drift.enabled: true`:
- `shutdown_step` (рекомендуется 40) — с этого шага non-novel illicit-схемы подавляются (аналог закрытия DarkMarket в Elliptic); `shutdown_rate_multiplier` — доля выживающих;
- `novel_step` (рекомендуется 45) + `novel_scheme` (wash) — новая схема, возникающая только на шагах 45–49 (проверка «обобщения на новом»).

Проверенный сценарий: `shutdown_step: 40`, `shutdown_rate_multiplier: 0.0`, `novel_step: 45` — wash строго на 45–49, non-novel illicit только до шага 40.

## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Снять статистики с реального Elliptic (data/raw/elliptic_txs_features.csv, ~690 МБ)
python -m kyt_engine._stats.compute

# 2. Сгенерировать датасет (configs/generator.yaml)
python -m kyt_engine.synth generate --config configs/generator.yaml

# 3. Проверить контракт Spillety
python -m kyt_engine.synth validate --dir data/synthetic/run
```

Детерминизм: одинаковый `seed` + одинаковый конфиг → байт-в-байт идентичные файлы (включая manifest).

## Структура

```
Hybrid-Theory/
├── .agents/skills/generator/SKILL.md  # навигационный SKILL по проекту
├── configs/generator.yaml          # параметры: размеры, схемы, дрифт
├── data/
│   ├── raw/                        # Elliptic (gitignored, источник статистик)
│   └── elliptic_stats/             # CDF-артефакты (создаёт compute)
├── src/kyt_engine/
│   ├── _stats/compute.py           # разовое снятие статистик с Elliptic
│   └── synth/
│       ├── config.py               # GeneratorConfig (yaml)
│       ├── stats.py                # загрузка CDF, sample_features (inverse-CDF+jitter)
│       ├── schemes.py              # шаблоны схем
│       ├── graph.py                # сборка графа, шагов, дрифта
│       ├── features.py             # фиче-матрица по классам
│       ├── emit.py                 # 3 CSV + manifest.json (sha256)
│       ├── validate.py             # контракт-валидатор (дубль spillety loader)
│       └── __main__.py             # CLI: generate | validate
├── tests/test_synth.py
└── TASKS/pivot-to-synthetic-data/  # QRSPI-артефакты
```

## Тесты

```bash
make test   # pytest tests/
make lint   # ruff check + format --check
```

## Лицензия

GNU AGPL — Copyright (c) 2026 Kirill Yuzhakov.