---
name: generator
description: Навигация по проекту Hybrid-Theory (генератор синтетических данных Elliptic++) — приоритетный порядок чтения репозитория, актуальные пути, контракт и критерии приёмки
---

Начинай изучение проекта с `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/README.md`.

## Актуальные, обязательные к чтению источники

- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/README.md` — что генерирует, CLI, структура
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/configs/generator.yaml` — размеры, схемы, дрифт (единственный источник параметров прогона)
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/__main__.py` — CLI: `generate | validate`
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/tests/test_synth.py` — как проверять генератор

## Модули генератора (что читать, если правишь конкретную часть)

- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/config.py` — `GeneratorConfig`/`DriftConfig` (yaml)
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/stats.py` — CDF-артефакты, inverse-CDF-сэмплинг, объёмы по шагам
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/schemes.py` — шаблоны схем: mixer, peel_chain, fanout, hub_spoke, wash (wash — дрифт-эксклюзивная novel-scheme)
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/graph.py` — сборка графа, роли/рёбра/time_step, temporal drift, `SchemeRun` (границы инстансов, без RNG)
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/features.py` — матрица фич feat_2..feat_166 по классу
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/anchors.py` — retrieval-слой: инстансы, edge-атрибуты (отдельный attr-RNG), якоря + K-hop, decoy-разметка, holdout-аннотация
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/emit.py` — 4 CSV + manifest.json (sha256 + ground_truth + retrieval-бенчмарк)
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/synth/validate.py` — дубль контракта загрузчика формата Elliptic++
- `https://github.com/KirillYuzh/Hybrid-Theory/blob/main/src/kyt_engine/_stats/compute.py` — разовое снятие статистик с реального Elliptic

## Контракт и критерии приёмки (генерация синтетики)

- `python -m kyt_engine.synth generate` -> 4 CSV + manifest.json: 10 000 tx, 14 092 рёбра
- Формат: features без header / classes с header / edgelist с header; `temporal_split` train ≤30, valid 31..40, test 41..49 -> 5856 / 2050 / 2094
- `python -m kyt_engine.synth validate` подтверждает контракт формата; классы — строки `{'unknown', '2', '1'}`
- Детерминизм: одинаковый seed -> байт-в-байт идентичные файлы
- Дрифт: novel-scheme (wash) появляется только на шагах 45–49; shutdown подавляет illicit до шага 40