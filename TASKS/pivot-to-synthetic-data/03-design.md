# 03-design.md — Synthetic Transaction Generator (K-BRAIN/TASKS/pivot-to-synthetic-data)

## Цель

Перестроить Hybrid-Theory из «KYT-движка с фейковыми метриками» в **генератор синтетических транзакций под контракт Spillety (Elliptic++)**. Никаких ML-моделей. Проект становится поставщиком данных/бенчмарков для Spillety: реалистичные фичи (маргинальные распределения из реального Elliptic), граф с посадкой известных схем (mixer, peel-chain, fan-out, hub-spoke, wash) и честные ground-truth лейблы + манифест.

## Зачем это Spillety (вклады)

1. **Ground-truth retrieval-бенчмарк**: знаем, какие tx входят в «схемы» и где «санкционные» узлы → можно точно мерить recall@10/PR-AUC HNSW/PQ.
2. **Аугментация редких классов**: посадка контролируемого числа известных схем (fan-out/peel-chain/mixer), которых нет в Elliptic.
3. **Якоря для контрастивного обучения**: «санкционированные» роли в манифесте → точные positive anchors.
4. **Стресс-тест temporal shift**: опциональный режим «закрытие darknet» / «появление ново-схемы» в поздних time_step → проверка drift-гейтов и walk-forward.
5. **Калибровка τ и операционные метрики**: известный базовый rate illicit → честный optimal τ, FP-нагрузка.

## Внешний контракт (равно `spillety/data/loader.py:6-45`)

Три файла в каталоге (например `data/synthetic/run1/`):

| Файл | Формат | Содержимое |
|------|--------|-----------|
| `elliptic_txs_features.csv` | **без header** | `txId, time_step, feat_2..feat_166` (167 колонок, 165 фич) |
| `elliptic_txs_classes.csv` | header | `txId, class`; class ∈ {1, 2, "unknown"} |
| `elliptic_txs_edgelist.csv` | header | `txId1, txId2` (рёбра вход→выход) |

Дополнительно:
- `manifest.json` — sha256 трёх файлов + конфиг генерации + ground-truth: список схем `{scheme, role, txIds[]}`, состав «санкционированных» роль/узлов, номер steps. (Аналог санкционного манифеста Spillety.)

## Реализм (уровень выбран пользователем)

- **Фичи**: маргинальные распределения `feat_2..feat_166` снимаются с реального `data/raw/elliptic_txs_features.csv` (разово, скриптом) и сохраняются как квантильные артефакты (эмпирический CDF на ~1000 точек на фичу × класс). Генератор делает inverse-CDF + джиттер.
  - Классы-источники: illicit (class 1), licit (class 2), background-unknown (class "unknown").
- **Граф**: шаблоны схем как подграфы (см. ниже), плюс фон случайных P2P-рёбер для связности (нужна для pagerank/степеней в Spillety).
- **Времена**: объём по шагам (tx/step) копируется из реального Elliptic (эмпирическое распределение counts/step). Tx одной схемы получают близкие time_step. Опционально: режим дрифта.

## Схемы (паттерны) — v1

| Схема | Структура | Роли | Класс |
|-------|-----------|------|-------|
| `mixer` | многие входы → mixer tx → многие выходы (fan-in/fan-out) | entry, mixer_core, exit | mixer_core = illicit, entry/exit по конфигу |
| `peel_chain` | линейная цепь A→B→C→…→N (последовательное отщепление) | chain_hop[i] | illicit |
| `fanout` | 1 scam tx → N жертв | scam, victim | scam=illicit, victim=licit |
| `hub_spoke` | рыночный/обменный хаб: многие tx → hub → многие tx | hub, spoke | hub=по конфигу |
| `wash` | цикл (cycle) | cycle_node[i] | illicit |
| `p2p` | фон: случайные пары tx, редкие рёбра | sender, receiver | licit |

Параметры каждой схемы (число узлов, разброс, веса) — в конфиге.

## Временной дрифт (опция)

- `shutdown_step` + `shutdown_rate_multiplier` — в шаге ≥ threshold базовый rate illicit снижается (аналог закрытия DarkMarket, шаг 43 в Elliptic).
- `novel_step` + `novel_scheme` — в шаге ≥ threshold прокидывается НОВАЯ схема, не встречавшаяся ранее (проверка «новых санкций/режимов»).
- Результат фиксируется в манифесте как `drift_scenario`.

## Сэмплинг фич

Для каждой tx по её роли выбирается источник-класс (illicit/licit/unknown). Из соответствующего артефакта для каждой из 165 фич: `F_i^{-1}(u) + jitter`, где u~U(0,1), jitter=малый гауссов по ширине локального бина CDF. Корреляции между фичами НЕ моделируются (осознанное упрощение — `# ponytail: marginal-only, add copula when Spillety needss joint realism`).

## Размеры и пропорции (конфиг)

- `n_txs` (def 40_000; Elliptic=203_769)
- `labeled_ratio` (def 0.23, как Elliptic 46k/203k)
- `illicit_ratio_in_labeled` (def 0.10, как Elliptic ~4.5k/46k)
- `unknown_ratio` = 1 - labeled_ratio (пополняют фон)
- Детерминизм: `seed`

## CLI

```
python -m kyt_engine.synth generate --config configs/generator.yaml --out data/synthetic/{name}
python -m kyt_engine.synth validate <dir>   # воспроизводит контракт loader.py
```

## Критерии приёмки (AC)

1. `validate` проходит точно такой же набор assert-ов, что `spillety/data/loader.py` (шapeы/колонки/значения классов, int time_step, без NaN).
2. Два прогона с одинаковым `seed` дают бит-в-бит одинаковые файлы.
3. Известные схемы, посаженные генератором, восстанавливаются из `manifest.json` (tx ↔ роль однозначны).
4. При `shutdown_step` значение precount illicit в поздних steps падает в соответствии с множителем.
5. Тесты зелёные: `make test`, `make lint`.

## Решено НЕ делать (YAGNI, v1)

- Моделирование корреляций между фичами (copula/GAN).
- Моделирование адресов/entity-ground-truth (разметка кластеров адресов) — отдельная итерация, зафиксировано в 05-plan как «позже».
- Встраивание FastAPI/demo.
- Интеграционный тест против реального Spillety (репо отдельное; реализован контрактный дубль loader в тестах).

## Comments

- нет открытых комментариев. (Были решены вопросы: направление — генератор; реализм — маргинальные фичи+граф+времена; старый код — удалить напрямую.)