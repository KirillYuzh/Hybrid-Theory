# Hybrid-Theory

Генератор синтетических криптовалютных транзакций в формате Elliptic++: по реальным статистикам блокчейн-транзакций строит вымышленный датасет, в котором заранее знаешь, что есть что. Это удобно для KYT/AML-экспериментов — поиска аномальных схем, обучения на ground truth, проверки устойчивости модели к временному дрейфу.

## Что получается на выходе

После прогона в `data/synthetic/run/` лежат четыре CSV и один JSON:

| Файл | Формат |
|------|--------|
| `elliptic_txs_features.csv` | без шапки: `txId, time_step, feat_2..feat_166` (167 колонок) |
| `elliptic_txs_classes.csv` | `txId, class`; class — строка: `1` (illicit), `2` (licit), `unknown` |
| `elliptic_txs_edgelist.csv` | `txId1, txId2` — рёбра графа транзакций |
| `elliptic_txs_edge_attributes.csv` | `txId1, txId2, amount, timestamp` — на каждое ребро сумма и время |
| `manifest.json` | конфиг, seed, sha256 всех файлов и вся «расшифровка»: какие схемы посажены, где якоря, какие суммы на рёбрах схем |

Эталонный прогон (`configs/generator.yaml`, seed 42): **10 000 tx, 14 092 ребра**. Временной сплит на привычные рубежи (train ≤30, valid 31–40, test 41–49) даёт **5856 / 2050 / 2094**; классы написаны как в настоящем Elliptic — `{'unknown', '2', '1'}`.

## Почему это выглядит правдоподобно

В `data/elliptic_stats/` лежат эмпирические распределения (квантильные сетки CDF) по каждой из 165 фич для трёх классов транзакций — их один раз снимают с настоящего Elliptic (203 769 tx) скриптом `kyt_engine._stats.compute`. Дальше генератор для каждой транзакции тянет признаки inverse-CDF-сэмплингом по её классу, так что синтетика повторяет статистику реальных данных, а не какие-то абстрактные случайности.

## Схемы-паттерны

Посаженные подграфы с ясной семантикой — это «правильные ответы» для тестов:

- `mixer` — fan-in -> mixer_core -> fan-out (ядро illicit);
- `peel_chain` — линейная цепь, все средние узлы illicit;
- `fanout` — scam -> жертвы (scam illicit);
- `hub_spoke` — обменный хаб (licit);
- `wash` — цикл wash-trading (illicit), **дрифт-эксклюзив**: появляется только в дрифт-режиме и только в конце временной шкалы;
- `p2p` — фон (licit/unknown + остаток illicit-бюджета): случайные направленные рёбра между обычными транзакциями, без гарантии связности.

## Пропорции (бюджеты)

`labeled_ratio` и `illicit_ratio_in_labeled` — это потолки, а не точные обязательства: схемы могут дать меньше размеченных, чем бюджет, а недобор добирается фоном. Жёстко гарантируется только одно — ровно `n_txs` транзакций.

## Retrieval-слой (включён по умолчанию)

Поверх схем строится контракт для поисковых бенчмарков:

- **Инстансы** — каждый посаженный подграф целиком: состав (`node_ids`/`edge_ids`), временное окно, точки сочленения, роли рёбер;
- **Якоря** — `per_1000_nodes` корней для запросов (жадный выбор с изоляцией ≥ `min_anchor_distance` хопов в полном графе) плюс `k_hop_neighborhoods` внутри инстанса;
- **Edge-attributes** — суммы и таймстампы рёбер с проверяемыми инвариантами: у mixer `sum(in) = sum(out) + fee`, у peel_chain сумма убывает, wash-цикл балансируется в пределах tolerance;
- **Decoys** — случайные fan_in/cycle-мотивы фона помечаются как «поисковый шум» (только аннотация, граф не трогается);
- **Holdout** — `entries: [{pattern, step_min, step_max, train_excluded}]` — разметка инстансов, попавших в нужное временное окно.

Суммы рёбер считаются своим отдельным RNG-потоком (`sha256(f"{seed}:attrs")`), поэтому эталонные числа выше от этого слоя не меняются, а выбор якорей и разметка полностью детерминированы.

## Temporal drift (опция)

`drift.enabled: true` имитирует знакомый по Elliptic эффект — «закрытие» крупного illicit-флоу и появление новой схемы в хвосте:

- `shutdown_step` (рекомендуется 40) — с этого шага старые illicit-схемы подавляются; `shutdown_rate_multiplier` — доля выживающих;
- `novel_step` (рекомендуется 45) + `novel_scheme` (wash) — новая схема на шагах 45–49 (проверка, обобщится ли модель на невиданное).

Проверенный сценарий: `shutdown_step: 40`, `shutdown_rate_multiplier: 0.0`, `novel_step: 45` — wash строго на 45–49, старый illicit только до шага 40.

## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Снять статистики с настоящего Elliptic (data/raw/elliptic_txs_features.csv, ~690 МБ)
python -m kyt_engine._stats.compute

# 2. Сгенерировать датасет (configs/generator.yaml)
python -m kyt_engine.synth generate --config configs/generator.yaml

# 3. Проверить, что датасет собран по контракту
python -m kyt_engine.synth validate --dir data/synthetic/run
```

Детерминизм: одинаковый `seed` + одинаковый конфиг -> байт-в-байт одинаковые файлы (включая manifest).

## Структура

```
Hybrid-Theory/
├── .agents/skills/generator/SKILL.md  # навигационный SKILL по проекту
├── configs/generator.yaml          # параметры: размеры, схемы, дрифт, retrieval
├── data/
│   ├── raw/                        # Elliptic (gitignored, источник статистик)
│   └── elliptic_stats/             # CDF-артефакты (создаёт compute)
├── src/kyt_engine/
│   ├── _stats/compute.py           # разовое снятие статистик с Elliptic
│   └── synth/
│       ├── config.py               # GeneratorConfig (yaml)
│       ├── stats.py                # загрузка CDF, sample_features (inverse-CDF+jitter)
│       ├── schemes.py              # шаблоны схем
│       ├── graph.py                # сборка графа, шагов, дрифта, SchemeRun (границы инстансов)
│       ├── anchors.py              # инстансы, edge-атрибуты, якоря, K-hop, decoy, holdout
│       ├── features.py             # фиче-матрица по классам
│       ├── emit.py                 # 4 CSV + manifest.json (sha256, ground truth, retrieval)
│       ├── validate.py             # контракт-валидатор
│       └── __main__.py             # CLI: generate | validate
├── tests/test_synth.py
└── TASKS/pivot-to-synthetic-data/  # артефакты задачи
```

## Тесты

```bash
make test   # pytest tests/
make lint   # ruff check + format --check
```

## Лицензия

GNU AGPL — Copyright (c) 2026 Kirill Yuzhakov.