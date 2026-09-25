# Обзор

## Что описано в этой версии

Hybrid-Theory здесь работает как генератор behavior-only. Граф транзакций строится только симуляцией поведения сущностей. Симуляция создаёт сущности, выбирает разрешённые действия и добавляет события в журнал, который не переписывается. Из этого журнала строятся Elliptic CSV и подробный `manifest.json`.

Версия имеет один режим работы:

- `graph.mode: behavior`;
- `features.mode: semantic`;
- канонический seed: `72`.

Проект не строит эмпирические распределения признаков и не использует случайный фоновый слой. В behavior-only нет отдельного фиксированного контракта на число рёбер и распределение классов. Размер набора задаёт `n_txs`, а число событий, рёбер и связей получается из расписания поведения.

## Как проходит запуск

1. Конфигурация проверяется до запуска генератора. Неизвестные поля, повторяющиеся ключи и неподходящие значения останавливают процесс.
2. Из `data/volume/volume.npy` берётся профиль объёма по 49 временным шагам. Он определяет распределение слотов, но не задаёт поведение отдельной транзакции.
3. Создаются типизированные сущности и запускается планировщик действий.
4. Каждое событие получает одну целевую транзакцию. Если у события есть причинный источник, добавляется направленное ребро от предыдущей транзакции.
5. Признаки и атрибуты рёбер вычисляются из получившегося графа.
6. Модуль записи сохраняет четыре CSV и `manifest.json`.
7. Отдельная проверка может сравнить синтетический набор с реальными файлами Elliptic через strict-inductive протокол.

## Файлы проекта

| Путь | Назначение |
|---|---|
| `configs/generator_behavior.yaml` | каноническая конфигурация behavior-only запуска |
| `src/kyt_engine/synth/config.py` | разбор и проверка YAML, defaults и сериализация effective config |
| `src/kyt_engine/synth/stats.py` | загрузка профиля `volume.npy` |
| `src/kyt_engine/synth/behavior.py` | профили, действия, сущности, события и причинный граф |
| `src/kyt_engine/synth/graph.py` | тонкий adapter behavior graph для эмиттера и downstream |
| `src/kyt_engine/synth/drift.py` | фазы временного расписания и распределение слотов |
| `src/kyt_engine/synth/features_semantic.py` | признаки из топологии, сумм, времени и chain metadata |
| `src/kyt_engine/synth/anchors.py` | явная проекция instances, anchors и k-hop neighborhoods |
| `src/kyt_engine/synth/emit.py` | запись четырёх CSV и behavior manifest |
| `src/kyt_engine/synth/validate.py` | проверка формата, ссылок и provenance |
| `src/kyt_engine/synth/features_structural.py` | общее структурное пространство для downstream |
| `src/kyt_engine/synth/gnn_downstream.py` | strict-inductive протокол, метрики и optional PyG |
| `src/kyt_engine/synth/__main__.py` | три CLI-команды behavior-only |
| `constraints-gnn.txt` | зафиксированные версии optional PyG environment |
| `data/volume/volume.npy` | входной профиль объёма по шагам |
| `data/raw/` | реальные CSV Elliptic для проверки transfer; генератор сюда не пишет |
| `data/synthetic/behavior_run/` | каталог канонического behavior run по умолчанию |

## Документы

- [Поведенческая модель](behavior-model.md)
- [Контракт данных](data-contract.md)
- [Strict-inductive downstream](downstream.md)
- [Конфигурация](configuration.md)
- [Разработка](development.md)

## Поддерживаемый CLI

В CLI behavior-only оставлены три команды.

### Сгенерировать набор

```bash
python -m kyt_engine.synth generate \
  --config configs/generator_behavior.yaml
```

Команда читает канонический YAML, строит поведенческий граф и записывает четыре CSV вместе с `manifest.json` в `out_dir`.

### Проверить набор файлов

```bash
python -m kyt_engine.synth validate \
  --dir data/synthetic/behavior_run
```

Проверяются схема Elliptic, хеши CSV, соответствие матрицы признаков, атрибутов рёбер, ссылок action/entity/event/instance и chain metadata.

### Проверить перенос на реальные данные

```bash
python -m kyt_engine.synth validate-real \
  --config configs/generator_behavior.yaml \
  --raw-dir data/raw \
  --out data/synthetic/behavior_run/strict_report.json \
  --tier smoke \
  --backend auto \
  --seeds 0 \
  --k 100
```

`validate-real` запускает отдельный протокол с фиксированной версией. Smoke run нужен для проверки механики и не является acceptance. Для real acceptance нужны закреплённый raw snapshot, зафиксированный протокол и окружение full PyG.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Нужны для downstream и полного PyG
pip install -e ".[downstream]"
pip install -c constraints-gnn.txt -e ".[gnn]"

python -m kyt_engine.synth generate --config configs/generator_behavior.yaml
python -m kyt_engine.synth validate --dir data/synthetic/behavior_run
```

Перед генерацией должен существовать `data/volume/volume.npy`. Это единственный числовой входной артефакт, который требуется поведенческому пути. Реальные CSV из `data/raw` нужны только для `validate-real`.

## Что важно помнить

- Seed `72` задаёт воспроизводимость самостоятельного behavior run. Набор seed `0..9` в strict downstream является отдельным набором экспериментов.
- Четыре CSV сохраняют Elliptic-форму, но не содержат всей информации о поведении. Для анализа сущностей и причинности нужен `manifest.json`.
- Семантические признаки не являются копией распределения Elliptic и не доказывают transfer quality.
- Full GraphSAGE, GAT и GIN являются optional backend. Их отсутствие не превращает diagnostic run в результат full GNN.
- Метрика `F1 > 0.5` в real acceptance является эмпирической целью протокола, а не гарантией результата.

Дальше по частям описаны модель поведения, формат данных, протокол downstream, конфигурация и рабочий процесс разработки.
