---
name: generator
description: Навигация по Hybrid-Theory: behavior-driven генератор, проверка данных и strict downstream
---

Начинай с `README.md`, затем читай `docs/overview.md` и `docs/behavior-model.md`.

## Основные файлы

- `configs/generator_behavior.yaml`: каноническая конфигурация, seed `72`.
- `src/kyt_engine/synth/behavior.py`: сущности, действия, события и причинные цепочки.
- `src/kyt_engine/synth/drift.py`: фазы временного расписания и распределение слотов.
- `src/kyt_engine/synth/features_semantic.py`: признаки из топологии, сумм и цепочек.
- `src/kyt_engine/synth/emit.py`: четыре CSV и manifest.
- `src/kyt_engine/synth/validate.py`: проверка формата, семантики и provenance.
- `src/kyt_engine/synth/gnn_downstream.py`: строгая индуктивная проверка на временных разделах.
- `tests/test_synth.py`: поведенческий контракт, детерминизм и отрицательные проверки.

## Команды

- `python -m kyt_engine.synth generate --config configs/generator_behavior.yaml`
- `python -m kyt_engine.synth validate --dir data/synthetic/behavior_run`
- `python -m kyt_engine.synth validate-real --config configs/generator_behavior.yaml --raw-dir data/raw --out data/synthetic/behavior_run/strict_report.json --tier smoke`

## Контракты

- Behavior run содержит ровно `n_txs` связанных событий и детерминированные CSV.
- `features.mode` всегда `semantic`.
- `unknown` не участвует в loss, threshold и метриках.
- Strict partitions: `1..30`, `31..40`, `41..49`; cross-partition edges удаляются.
- Full backend не заменяется SGC молча. SGC и auto без optional PyG являются диагностическими.
- Real acceptance требует pinned raw snapshot, canonical config и frozen seeds `0..9`.

## Ограничения

- Семантические признаки не воспроизводят joint distribution Elliptic.
- P0 chain metadata не является полным bridge или asset ledger.
- Event ledger не доказывает полное сохранение mixer или round-trip motif.
- `F1 > 0.5` является целью эксперимента, а не гарантией.
