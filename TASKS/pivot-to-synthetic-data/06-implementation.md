# 06-implementation.md — Реализация (фаза I)

Статус: **завершено**, 0 🔴 Critical, 0 ai-slops (raном), AC выполнены.

## Что изменено

### src/kyt_engine/synth/

- **graph.py** — переписан:
  - `wash` (novel_scheme) никогда не попадает в regular-пул; novel-проход — только при `drift.enabled`. shuffle(novel) сохраняет стабильность RNG-потока (рёбра/сплит не меняются между прогонами).
  - гейт дрифта по `birth + OFFSET_RANGE - 1` (последний возможный шаг узлов схемы), а не по `birth` — закроd утечку illicit на шагах ≥ shutdown_step.
  - `_register` чище: шаги считаются до создания узлов, без мёртвой первой записи и слайс-«угадайки».
  - фон: бюджеты — «не больше бюджета» (relaxed), точное n_txs гарантировано; `n_bg < 0` → явный `ValueError` вместо краша `np.full`.
  - комментарий о p2p-рёбрах честный: случайные пары, без гарантии связности.
- **stats.py** — удалены мёртвые `LABEL_TO_CLASS`/`CLASS_TO_SOURCE`/`step_weight`/`STEP_MAX`; добавлен единый `LABEL_TO_CLASS` (raw→class) и `CLASS_TO_LABEL` (class→"1"/"2"/"unknown").
- **schemes.py** — `fanout`/`hub_spoke` свёрнуты в один параметризованный `_star`; убран docstring-slop.
- **features.py** — `N_FEATURES` из stats вместо литерала `2+165`; убран неверный docstring («по классу роли» → «по классу»).
- **emit.py** — удалён дублирующий параметр `rng_seed`; mapping классов из `CLASS_TO_LABEL`.
- **config.py** — дефолты из инстанса (один источник правды), гит режимные значения дрифта по AC: `shutdown_step: 40`, `shutdown_rate_multiplier: 0.0`, `novel_step: 45`; `to_dict` снова включает `stats_dir`.
- **validate.py** — `assert` заменены на явные `ValueError` (работает под `python -O`); убран тавтологичный ассерт; `load_elliptic` больше не читает файлы повторно.
- **__main__.py** — `write_dataset` без лишнего аргумента; полная проверка артефактов статистик (все CDF + volume).

### Конфиг и README

- **configs/generator.yaml** — значения дрифта по AC (40 / 0.0 / 45, `enabled: false`); комментарии о бюджетах, p2p, wash.
- **README.md** — честные схемы (peel_chain = средние узлы; p2p без гарантии связности; wash = дрифт-эксклюзив), формат Elliptic++, эталонные числа AC, раздел про бюджеты и дрифт.

### Тесты (tests/test_synth.py)

- `test_deterministic` — теперь настоящий: отдельные out_dir, сравнение байт-в-байт всех 4 файлов (включая manifest).
- `test_wash_only_as_drift_scheme` — wash отсутствует при drift off, присутствует (45–49) при drift on.
- `test_drift_acceptance` ×6 сидов — рубежи AC: wash ⊆ 45..49, non-wash illicit < 40.
- `test_drift_reduces_late_illicit` — сохранён под новый гейт.
- убрана прослойка `stats_dir_path`; фикстура `stats_dir`.

## Итоговая верификация

| AC | Значение |
|---|---|
| generate → 10 000 tx / 14 092 edges | ✅ |
| temporal_split 5856 / 2050 / 2094 | ✅ |
| validate + классы-строки {'unknown','2','1'} | ✅ |
| детерминизм byte-identical (2 прогона CLI, diff пуст) | ✅ |
| дрифт: wash 45–49, non-wash illicit < 40 (сиды 0..19) | ✅ |

`make test` — 11 passed, `make lint` — clean.

## Решения пользователя
- SKILL → `.agents/skills/generator/SKILL.md`.
- Дрифт: конфиг 40 + гейт по offset (рекомендация).
- labeled_ratio: бюджет «не больше» (relaxed).
- wash: дрифт-эксклюзив (novel), в drift-off не генерируется; числа AC сохранены (14 092 / 5856:2050:2094).