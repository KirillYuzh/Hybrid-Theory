# Review: retrieval-bench слой синтетического генератора (diff поверх 790b03b)

Дата: 2026-09-22
Объект: `src/kyt_engine/synth/{anchors,graph,config,emit,validate,__main__}.py`, `configs/generator.yaml`, `tests/test_synth.py`
Метод: code-review (K-BRAIN/SKILLS/code-review.md) + ai-slops (K-BRAIN/SKILLS/ai-slops.md)

## Что проверено (команды и результаты)

- Прочитаны полностью: `anchors.py` (452 стр.), `graph.py`, `config.py`, `emit.py`, `validate.py`, `__main__.py`, `configs/generator.yaml`, `tests/test_synth.py`, плюс контекст `schemes.py`, `stats.py`, дизайн-контракт `TASKS/.../07-retrieval-bench-design.md`.
- `.venv/bin/python -m pytest tests/ -q` → **16 passed** (11 старых + 5 новых).
- `.venv/bin/python -m ruff check src tests` → **All checks passed**.
- `.venv/bin/python -m ruff format --check src tests` → **14 files already formatted**.
- Живой прогон дефолта (`configs/generator.yaml`): `Generated 10000 txs, 14092 edges` — AC-числа сохранены.
- Живая проверка манифеста дефолта: anchors = **20/20** (target 20), 85 инстансов; pairwise undirected BFS по 190 парам якорей — none ближе `min_anchor_distance=3`; decoy: 1598 fan_in + 1 cycle.
- Прогон c `per_1000_nodes: 20` (target 200): якорей 85 = число инстансов (макс. потолок — «не добираем»).
- Проверен floor-случай peel (см. 🟡 S1) и дублирование замыкающего узла в cycle-decoy (см. 🟡 S4).

## AI-slops: что выкинуть / упростить

### S-slop-1. Мёртвое поле `PatternInstance.decoy` + мёртвые guards
`anchors.py:38` (`decoy: bool = False`), `anchors.py:336` (`if inst.decoy: continue` в `select_anchors`), `anchors.py:442` (`if inst.decoy: continue` в `apply_holdout`).

Крит-идея дизайна (дизайн-док §2.1) — decoy как отдельный `PatternInstance` с `decoy=True`. В реализации decoy — простые dict из `detect_decoys` (emit.py:62) и **никогда** не становятся `PatternInstance`. Значит поле и обе guard-ветки — мёртвый код, недостижимый никогда.

Решение: удалить поле `decoy` (или оставить только если decoy реально станет инстансом); удалить `if inst.decoy: continue` в обоих функциях. Обоснование: ai-slops п.3 «мёртвые поля/методы», критерий выхода «0 ai-slops».

### S-slop-2. Неиспользуемый параметр `config` в `_instance_entry`
`emit.py:146` — `def _instance_entry(inst, config: GeneratorConfig)` — `config` в теле не используется.

Решение: убрать параметр, поправить вызов (emit.py:81). Обоснование: мёртвый параметр, путает читателя (создаёт иллюзию, что функция зависит от конфига).

### S-slop-3. Повтор dict-литерала атрибутов ребра ×6
`anchors.py:217-222, 223-230, 236-241, 253-258, 266-271, 275-280, 284-289` — одинаковый литерал `{"txId1": u, "txId2": v, "amount": _to_amount(c), "timestamp": timestamp(u, v)}` в **7** копиях.

Решение: вынести helper `_edge_attrs(u: int, v: int, amount: int) -> dict` (или `amount` уже в «долларах»), три строки вместо шести блоков. Обоснование: сейчас любое изменение контракта CSV (новое поле, переименование) — правка в 6 местах, риск рассинхрона.

### S-slop-4. Неверный тип `attrs: list[dict]`
`anchors.py:185` — `attrs: list[dict] = [None] * len(graph.edges)` — список на самом деле `list[dict | None]` (заполняется во второй фазе).

Решение: `attrs: list[dict | None] = [None] * len(graph.edges)`. Обоснование: точность типов; сейчас mypy strict (pyproject.toml:37) это бы поймал.

### S-slop-5. Docstring-пересказы имён функций
`anchors.py:124` «Derive per-instance annotations from scheme_runs» и `anchors.py:330` «Greedy anchor selection by instance order» — пересказ имени по comments.md («нельзя пересказывать имя»). Полезная часть — контракт *без RNG* в комментарии.

Решение: сократить до контракта, напр. `build_instances` → «Zero RNG. node/edge ids по схеме run-блоков» (или слить в коммент над ATTR_SALT). `select_anchors` — оставить только «No RNG; greedy isolation »min_distance; бюджет target — потолок, не добираем». Обоснование: comments.md.

### S-slop-6. Дублирование BFS/adjacency между тестами и продакшеном
`tests/test_synth.py:148-171` (`_bfs`, `_undirected`) vs `anchors.py:293-318` (`_bfs_distance`, `_full_adjacency`) — фактически копии.

Замечание как decision-point: mirror-реализация в тесте — осознанная практика (тест НЕ тавтологичен, ловит баг в продакшене), поэтому это не однозначный slop, но против правил «без задвоений». Комментарий `_bfs` сам признаёт «Mirrors anchors._bfs_distance». Решение: оставить mirror, но добавить `# ponytail:`-пометку (осознанное дублирование ради независимости проверки) ИЛИ импортировать из `anchors` (тогда тест станет тавтологией — хуже). Рекомендуется первый вариант.

### S-slop-7. Прочие мелочи
- `generator.yaml` — нет trailing newline (diff «no newline at end of file»), приадить.
- `emit.py:66` коммент `# indexes == edgelist row numbers` — ок, контракт; не трогать.

---

## 🔴 Critical

**Нет.** Все AC-инварианты, перечисленные в ТЗ, подтверждены живым прогоном и тестами:
- 10000 tx / 14092 edges / byte-детерминизм — совпадают (тесты + живой прогон);
- `_register`/`SchemeRun` (graph.py:79-110) — только чтение счётчиков, 0 RNG;
- `edge_id == индекс edgelist == индекс attrs` — корректно по построению (graph.py:82-83) и проверено `validate_edge_attributes`;
- mixer `sum(in)==sum(out)+fee` — точное (последний out поглощает остаток, anchors.py:212);
- wash per-node `|in-out| <= tol*max` — гарантировано clamp-ом кумулятивного дрейфа (см. ✅);
- 16/16 тестов, ruff clean;
- якоря 20/20, изоляция соблюдена, `dist=None` (кап) не отвергается.

---

## 🟡 Suggestions

### S1. Peel: `max(1, int(amount * drip))` ломает «strictly decreasing» на флор
`anchors.py:242-244`. При `amount == 1`: `1 * drip` → `int() == 0` → `max(1, 0) == 1` — ряд перестаёт строго убывать. Легальная конфигурация (`start_amount: [1,2]`, `length_range: [4,8]`, drip `[0.5,0.75]`) даёт последовательность `2, 1, 1, 1, 1` — нарушение инварианта (проверено симуляцией). Дефолт безопасен (min_end ≈ 5000·0.5⁶ ≈ 78).

Решение (вариант): документировать условие использования в конфиге, либо менять формулу на строгое убывание до флора и зафиксировать инвариант манифеста как «non-increasing с гарантией убывания пока > 1», либо в `validate_edge_attributes`/тестах проверять `amounts[i] > amounts[i+1] or amounts[i] == 1`. Обоснование: инвариант заявлен как контракт бенчмарка, но не защищён от пользовательского YAML.

### S2. Decoy cycle: в `nodes` дублируется замыкающий узел
`anchors.py:411` — `cycle = path[start:] + [v]`: для простого цикла v0→v1→v2→v0 в манифесте `nodes: [v0, v1, v2, v0]` (5 записей) при 4 рёбрах — на живом прогоне подтверждено. Итоговые данные слегка странные для потребителя (count(nodes) != count(edges)+1).

Решение: `cycle = path[start:]` (без дубля) — рёбра и так образуют замкнутый цикл; либо явно документировать «замыкающий узел повторён». Обоснование: качество данных манифеста; тест `test_decoys_valid` сейчас не ловит (проверяет только motif и scheme).

### S3. `validate_edge_attributes` не реализует обещанные дизайном инварианты
`validate.py:70-86` — проверяет только структуру (число строк, колонки, txId совпадение, amount >= 0). Дизайн-док §10 (p.332) обещал «invariants п.4.3 по edge_attribute_ground_truth» (mixer fee, peel убывание, wash баланс). Фактически инварианты живут только в `test_edge_attribute_invariants`.

Решение: либо добавить инварианты (потребует manifest + атрибутный GT), либо явно пометить в доке «инварианты проверяются тестом, а не validate». Обоснование: расхождение дизайн-контракта и реализации; сегодня `validate` пропустит сломанные суммы при «чистых» колонках.

### S4. `from_yaml`: wholesale-replace `amount` без мержа с дефолтами
`config.py:110` — `edges.get("amount", defaults.edge_attributes.amount)`. Если пользователь задаст `amount` частично (только mixer), то `build_edge_attributes` получит KeyError на `amount_conf["p2p"]` (anchors.py:193 и :287). Аналогично `entity_types` (config.py:120). Немердженные конфиги `from_yaml` не дают никакого контракта полноты.

Решение: мёржить с дефолтами по ключам (`{**defaults, **user}` рекурсивно для amount), либо валидировать обязательные ключи `{mixer, peel_chain, wash, hub_spoke, fanout, p2p}` в `from_yaml`. Обоснование: сейчас легальный YAML роняет генератор нетипичным KeyError.

### S5. `k_hop_range: []` → `max()` бросает ValueError
`anchors.py:351` — `max(config.anchors.k_hop_range)` падает на пустом списке, и при `max_anchor_distance=0` (вырожденный инстанс) `range(1, 1)` вернёт пустой dict — ок, но пустой `k_hop_range` — крэш.

Решение: `max(config.anchors.k_hop_range or [inst.max_anchor_distance])` или валидация в конфиге. Обоснование: робастность к пользовательскому конфигу.

### S6. `retrieval_specs`: самописание противоречит формату CSV
`emit.py:128-137`. `features.csv` пишется без заголовка (`header=False`, emit.py:46), а `node_id_column: "txId"` и `node_vector_lookup: "row of vectors_file where txId column == node id"` подразумевают колонку. Поле `default_k: 10` — произвольная константа, нигде не используется и расходится с `k_hop_range: [1,3]`.

Решение: переписать lookup как «row = txId (первая колонка без заголовка) == node id», либо ввести заголовки; `default_k` либо убрать, либо вывести из k_hop_range. Обоснование: машиночитаемость манифеста как контракта бенчмарка.

### S7. Тесты: дублирование проверок и magic-константы
- `tests/test_synth.py:174-181` повторяет то, что уже делает `validate_edge_attributes` (rows/len/txId1) — redundant.
- `tests/test_synth.py:247` `tol = 0.05` — продублировано из конфига `balance_tolerance`; :241 `0.004..0.021` — из fee_fraction. При правке конфига тесты молча отстанут.

Решение: брать `tol`/границы из `cfg.edge_attributes.amount[...]`; убрать дублирующие пары assert (оставить `validate_edge_attributes` как единственный источник структурной проверки). Обоснование: тесты должны читать конфиг, а не зашивать его значения.

### S8. Хрупкость `apply_holdout` к пропущенным ключам
`anchors.py:445-448` — `entry.get("pattern")` + прямой `entry["step_min"]`/`entry["step_max"]`: при записи без `step_min` — KeyError, не `windows_active=False`. Также `windows_active` = True даже если записи есть, но ни один инстанс не попал в окно (легально при `pattern: wash` с drift.off).

Решение: `entry.get("step_min")`/`entry.get("step_max")` + пропуск неполных записей, либо валидация структуры entries в `from_yaml`. Обоснование: контракт holdout-разметки сейчас зависит от «правильного» YAML без проверки.

### S9. `to_dict` не сериализует `out_dir` (pre-existing, не от этого диффа)
`config.py:137-169` — `out_dir` и `stats_dir` есть, а `out_dir` в `to_dict` нет, при этом `from_yaml` его читает. Манифест получает неполный конфиг.

Решение: добавить `"out_dir": str(self.out_dir)` в `to_dict`. Обоснование: симметрия round-trip конфига и полнота манифеста. (Мелочь, pre-existing.)

### S10. Тип `EdgeAttributeConfig.amount`: shallow copy разделяет вложенные list
`config.py:39` — `dict(DEFAULT_EDGE_AMOUNTS)` копирует только верхний уровень; вложенные `[lo, hi]` общие. Сейчас никто не мутирует — безопасно, но при будущей правке конфига из кода — утечка между инстансами.

Решение: `copy.deepcopy(DEFAULT_EDGE_AMOUNTS)`. Обоснование: дешёвая защита от классического aliasing-бага.

### S11. Потенциальное дублирование в `kw_hop`/`_eccentricity`/`_bfs_distance` — три BFS в одном модуле
`anchors.py:60-71` (`_eccentricity`), :293-308 (`_bfs_distance`), :347-360 (`k_hop_neighborhoods`) — три независимых BFS с одинаковым паттерном `dist={...}; queue=[...]; for u in queue` над разными adjacency.

Решение: общий `_bfs_layers(adj, src, cap)` возвращающий dist-словарь (или None-семантику), от которого выразить все три. Обоснование: ai-slops п.5 «лишние прослойки» — наоборот, здесь дублирование; консолидация убирает 3 копии алгоритма. (Оставить, если вкусовое.)

---

## ✅ Good Practices

- **AC-сохранность выполнена правильно.** Атрибутный RNG изолирован: `sha256(f"{seed}:attrs")[:8]` → `default_rng` (anchors.py:14-22), ровно по дизайну R2; `_register`/`SchemeRun` не тянут RNG; живой прогон дал 10000/14092, `test_deterministic` — byte-identical все 5 файлов + manifest.
- **Жёсткое выравнивание индексов.** node/edge ids инстанса выводятся из contiguous run-блоков (graph.py:82-100), `attrs[k]` заполняется по `edge_id` == номер строки edgelist; `validate_edge_attributes` это закрепляет (validate.py:78-83).
- **Mixer-инвариант точный, не приближённый.** fee = `round(total_in·f)`, out-бюджет делится взвешенно, последний out поглощает остаток → `sum(in) == sum(out) + fee` в центах (anchors.py:198-212).
- **Wash: clamp кумулятивного дрейфа реально гарантирует баланс в допуске.** `new_cum = clamp(cum·r, 1/(1+tol), 1+tol)` (anchors.py:247-263): это ограничивает не только шаг, но и произведение, поэтому для каждого узла `|in-out| ≤ tol·max(in,out)` — я проверил границы аналитически (r_max = hi/1.0 = 1.05, lo-сторона 0.0476 < 0.05), замыкающее ребро возвращает `base` и балансирует узел-0 точно.
- **Isolation-семантика BFS-капа корректна.** `_too_close` = «measured d < min», `None` (дальше капа) не отвергает (anchors.py:321-324) — живо проверено на 190 парах: ни одной пары ближе 3.
- **Детерминизм везде, где не RNG:** отсортированные итерации в `_instance_adjacency`, `_full_adjacency`, decoy (sorted `in_sources`, sorted(adj), sorted(srcs)), Tarjan; `select_anchors` идёт по instance_id.
- **`validate` CLI backward-compatible:** FileNotFound → аккуратное учтивое сообщение для старых датасетов (__main__.py:40-45).
- **Тесты покрывают контракт, а не только «не падает»:** fee-границы, строгое убывание peel, wash-баланс по каждому узлу, изоляция якорей независимым BFS, decoy-мотивы, holdout-разметка, forward/backward совместимость manifest (`instance_id` добавлен).

---

## Итог

- 🔴 Critical: **0**
- 🟡 Suggestions: **11** (S1–S11) + 7 ai-slop-пунктов (S-slop-1..7, из них S-slop-1 — реальный мёртвый код)
- Файлы с проблемами: `src/kyt_engine/synth/anchors.py`, `src/kyt_engine/synth/emit.py`, `src/kyt_engine/synth/config.py`, `src/kyt_engine/synth/validate.py`, `tests/test_synth.py`, `configs/generator.yaml`

Вывод: слой принимается по AC, но перед «LGTM» — убрать мёртвый `PatternInstance.decoy` и неиспользуемый параметр `_instance_entry` (ai-slops), и решить S1/S2/S4 (инварианты при пользовательских конфигах).