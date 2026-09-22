# 04 Subagent — Контракт Spillety vs генератор: проверка AC

Отчёт subagent-4 (сверка с реальным `spillety/data/loader.py`). Здесь тезисы; полный разбор в сессии.

## 1. Реальный контракт Spillety (`spillety/data/loader.py`, ветка main)

`load_elliptic(root)` внутри assert-ов НЕ имеет; контракт — docstring + тип возврата `(features, classes, edgelist, merged)`. Проверки — только в `__main__` (реальный Elliptic: 203769 tx, 234355 edges, 203769×168; temporal 123287/38316/42166).

Чтение:
- features: `pd.read_csv(..., header=None)` → 167 колонок `txId,time_step,feat_2..feat_166`, затем `astype(int)` для time_step;
- classes: `pd.read_csv(...)` с заголовком `txId,class`;
- edgelist: с заголовком `txId1,txId2`;
- `merged = features.merge(classes, on="txId", how="inner")` → 168 колонок.

`temporal_split(df, time_col="time_step", train_end=30, valid_end=40)`:
- train: `time_step <= 30`;
- valid: `31..40` (шаг 40 входит в valid);
- test: `> 40` (с 41).

(Spillety в venv не установлен — контракт воспроизведён теми же выражениями pandas, что в loader.py.)

## 2. Таблица AC

| AC | Статус | Обоснование |
|---|---|---|
| 1. generate → 3 CSV + manifest; 10 000 tx / 14 092 edges | ✓ | Реальный запуск: `Generated 10000 txs, 14092 edges`; формы/заголовки — `emit.py:33-68`. |
| 2. load_elliptic + temporal_split → 5856/2050/2094 | ✓ | Воспроизведение 1:1: shapes (10000,167)/(10000,2)/(14092,2)/(10000,168); сплит 5856/2050/2094 (сумма 10000). |
| 3. validate + классы-строки {'unknown','2','1'} | ✓ | validate прошёл; колонка читается dtype `str`, counts {'unknown':7700,'2':2070,'1':230}. Причина: значения в dict — строки (`emit.py:43-45`); 'unknown' не даёт приведения к числу. |
| 4. Детерминизм byte-identical | ✓ | Два прогона seed 42 в разные каталоги — все 4 файла IDENTICAL. Методетатерминизм: нет sort_keys, но порядок детерминирован (вставка, порядок генерации). |
| 5. Drift: wash только 45..49; shutdown без non-novel illicit от 40 | ✗ | Текущий конфиг: enabled:false, shutdown_step:44, mult:0.2, novel_step:46. Не выполняется ни одно условие. |

## 3. AC#5 — минимальный дифф

- **wash 45..49**: нужен `novel_step: 45` (birth 45..49, offset-клип 49). Сейчас 46 → wash 46..49.
- **shutdown**: гейт `graph.py:99` по `birth`, но узлы выходят на `birth+offset(0..2)` → при shutdown_step=40/mult=0.0 по 200 сидам утечки на шагах 40 (360 узлов) и 41 (165 узлов) из birth 38/39. Фон корректно ограничен шагом ≤ shutdown_step−1 (`graph.py:139-142`).

Два варианта:
1. Только конфиг: `shutdown_step: 38`, `mult: 0.0`, `novel_step: 45` → max non-wash illicit ≤ 39 (проверено 200 сидов). Минус: семантика «38» против «до 40».
2. Дифф `graph.py:99-100`: гейт по offset-скорректированному последнему шагу (`last = clip(birth+2)`), конфиг `shutdown_step: 40`, `mult: 0.0`, `novel_step: 45` — честные «40». Рекомендуется.

Плюс конфиг: `drift.enabled: true`, `shutdown_rate_multiplier: 0.0`.

## 4. Риски
- wash полностью illicit — тест «нет illicit после 40» обязан исключать non-wash/non-novel.
- На 45..49 законно есть hub_spoke/p2p — проверять subset-свойство wash, а не «ничего кроме wash».
- Для финальной приёмки AC#2 стоит `pip install` репозитория Spillety и прогнать реальные `load_elliptic`/`temporal_split` (в venv пакета нет).
- `validate` проверяет только формат (AC#3), дрифт (AC#5) — отдельным тестом по ground_truth.
- Edgelist пишется С заголовком (`emit.py:51`) — осознанное подчинение загрузчику Spillety (реальный Elliptic-файл без заголовка).