# 05-plan.md — План реализации

## Предварительно: обработка статистик из Elliptic (один раз)

1. Написать `src/kyt_engine/_stats/compute.py`: читает `data/raw/elliptic_txs_features.csv` (chunked, header=None) + classes; для каждого класса (1/2/unknown) и каждой из 165 фич строит эмпирический CDF (~1000 квантилей) → `data/elliptic_stats/cdf_{class}.npy`; counts/step → `data/elliptic_stats/volume.npy`; опционально `step_to_default_step.csv` нет. Запустить, чтобы артефакты были в рабочем дереве (ускорение дальнейших итераций).
   - **Пушбэк (риск)**: файл 689МБ, построчный chunked считываем без numpy нагрузки — O(10 сек) на pandas; следить память.

## План кода (порядок)

### Под-шаг 1: каркас + конфиг + контракт-валидатор
- `pyproject.toml`, `Makefile`, `configs/generator.yaml`, `src/kyt_engine/synth/{config,validate,__main__}.py`, `tests/test_contract.py`.
- `validate.py` воспроизводит asserts `spillety/data/loader.py` (шape (n,167), time_step int, class∈{1,2,unknown}, edgelist 2 колонки, no NaN).
- Ревью code-review+ai-slops.

### Под-шаг 2: статистики + сэмплинг фич
- `_stats/compute.py` + `synth/stats.py` (inverse-CDF + jitter).
- Тест: сгенерированные фичи ~ той же причиной (средние/процентиль совпадают в допусках) — `tests/test_contract.py` или отдельный `test_stats.py`.

### Под-шаг 3: схемы + граф + время
- `synth/schemes.py` (mixer, peel_chain, fanout, hub_spoke, wash, p2p), `synth/graph.py` (роли, рёбра, time_step по volume + дрифт).
- `tests/test_schemes.py` (AC-3), `tests/test_drift.py` (AC-4).

### Под-шаг 4: эмиттер + manifest
- `synth/features.py`, `synth/emit.py` (3 CSV + manifest.json с sha256 и схемами).
- Обновить README.

### Под-шаг 5: интеграция
- `python -m kyt_engine.synth stats/generate/validate` end-to-end на маленьком прогоне; `make test`, `make lint`.
- Ревью code-review+ai-slops, критерии AC из 03-design.

## Позже (не сейчас)
- Адресная модель + entity-resolution ground truth (кластеры адресов) для ERR-тестов Spillety.
- Совместные (copula/GAN) фичи.