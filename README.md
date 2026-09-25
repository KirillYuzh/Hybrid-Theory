# Hybrid Theory

Генератор синтетических криптовалютных транзакций для исследований KYT и AML. Он создаёт граф, в котором видны сущности, действия, связи и изменения поведения во времени.

## Что делает проект

Генератор создаёт семантические признаки, направленные рёбра, суммы, временные шаги и подробный manifest с происхождением каждого события.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m kyt_engine.synth generate --config configs/generator_behavior.yaml
python -m kyt_engine.synth validate --dir data/synthetic/behavior_run
```

Для проверки на реальных данных нужен каталог `data/raw`:

```bash
python -m kyt_engine.synth validate-real \
  --config configs/generator_behavior.yaml \
  --raw-dir data/raw \
  --out data/synthetic/behavior_run/strict_report.json \
  --tier smoke
```

## Что находится в результате

В каталоге появляются четыре CSV и `manifest.json`. В manifest записаны сущности, события, цепочки, фазы поведения, chain metadata, настройки RNG и хеши файлов.

## Документация

- [Обзор проекта](docs/overview.md)
- [Поведенческая модель](docs/behavior-model.md)
- [Формат данных](docs/data-contract.md)
- [Strict downstream](docs/downstream.md)
- [Конфигурация](docs/configuration.md)
- [Разработка и проверки](docs/development.md)

## Ограничения

Семантические данные не воспроизводят статистику Elliptic один в один. Chain metadata описывает синтетические идентификаторы и логические переходы, но не является полным bridge ledger. Значение `F1 > 0.5` остаётся целью эксперимента, а не гарантией.

## Лицензия

GNU AGPL, Copyright 2026 Kirill Yuzhakov.
