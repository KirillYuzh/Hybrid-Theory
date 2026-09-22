---
name: generator
description: Навигация по проекту Hybrid-Theory (генератор синтетических данных Elliptic++ для Spillety) — приоритетный порядок чтения репозитория, актуальные пути, контракт и критерии приёмки
---

Начинай изучение проекта с `K-BRAIN/README.md` (правила работы с этим репо), затем с `README.md` в корне проекта.

## Актуальные, обязательные к чтению источники

- `README.md` — что генерирует, CLI, структура
- `configs/generator.yaml` — размеры, схемы, дрифт (единственный источник параметров прогона)
- `src/kyt_engine/synth/__main__.py` — CLI: `generate | validate`
- `tests/test_synth.py` — как проверять генератор

## Детальное описание, может устаревать

- `TASKS/pivot-to-synthetic-data/` — QRSPI-артефакты текущей задачи (design/structure/plan)
- `TASKS/pivot-to-synthetic-data/artifacts/subagents/` — отчёты subagents (root-cause старого кода, анализ Spillety, ai-slops-ревью, сверка с контрактом)

## Модули генератора (что читать, если правишь конкретную часть)

- `src/kyt_engine/synth/config.py` — `GeneratorConfig`/`DriftConfig` (yaml)
- `src/kyt_engine/synth/stats.py` — CDF-артефакты, inverse-CDF-сэмплинг, объёмы по шагам
- `src/kyt_engine/synth/schemes.py` — шаблоны схем: mixer, peel_chain, fanout, hub_spoke, wash (wash — дрифт-эксклюзивная novel-scheme)
- `src/kyt_engine/synth/graph.py` — сборка графа, роли/рёбра/time_step, temporal drift
- `src/kyt_engine/synth/features.py` — матрица фич feat_2..feat_166 по классу
- `src/kyt_engine/synth/emit.py` — 3 CSV + manifest.json (sha256 + ground_truth)
- `src/kyt_engine/synth/validate.py` — дубль контракта `spillety/data/loader.py`
- `src/kyt_engine/_stats/compute.py` — разовое снятие статистик с реального Elliptic

## Полезные ссылки

- `https://github.com/Spillety/Spillety/blob/main/README.md`
- `https://github.com/Spillety/Spillety/blob/main/data/loader.py` — целевой контракт формата (features без header / classes с header / edgelist с header; `temporal_split` train ≤30, valid 31..40, test 41..49)
- `https://github.com/Spillety/Spillety/blob/main/.agents/skills/spillety/SKILL.md` — SKILL-навигация по Spillety

## Контракт и критерии приёмки (генерация синтетики)

- `python -m kyt_engine.synth generate` → 3 CSV + manifest.json: 10 000 tx, 14 092 рёбра
- Реальный `spillety.data.loader.load_elliptic` читает синтетику без ошибок; `temporal_split` даёт 5856 / 2050 / 2094
- `python -m kyt_engine.synth validate` подтверждает контракт Spillety; классы — строки `{'unknown', '2', '1'}`
- Детерминизм: одинаковый seed → байт-в-байт идентичные файлы
- Дрифт: novel-scheme (wash) появляется только на шагах 45–49; shutdown подавляет illicit до шага 40

## Связка с K-BRAIN

- Код-ревью: `K-BRAIN/SKILLS/code-review.md` + `K-BRAIN/SKILLS/ai-slops.md` после каждого под-шага
- Разработка: `K-BRAIN/SKILLS/development.md` (ленивый сеньор, anti-overengineering)
- Комментарии: `K-BRAIN/SKILLS/comments.md` (на английском, только сложная логика)
- Workflow: `K-BRAIN/WORKFLOW.md` (QRSPI) — артефакты в `TASKS/<slug>/`
- Задачи: `K-BRAIN/TODO.md` — вести по мере выполнения