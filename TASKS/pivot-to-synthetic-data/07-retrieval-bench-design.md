# 07-retrieval-bench-design.md — Дизайн-контракт: anchors / retrieval / edge-attributes / holdout

Статус: **дизайн** (фаза I, артефакт для реализации). Кода нет.
Критический инвариант: **AC-сохранность** — эталон `10000 tx / 14092 edges / temporal split 5856:2050:2094`, byte-identical детерминизм при том же seed, wash только как drift-novel — **не меняются** при дефолтном конфиге.

---

## 1. Цель

Следующий слой после `06-implementation`: дать Spillety бенчмарк retrieval с **точками-якорями**, **структурными ground-truth** (кто в схеме, кто сосед якоря на каждом хопе), **атрибутами рёбер** (суммы/инварианты) и **decoy-подграфами** в p2p-фоне. Всё — аннотации поверх уже сгенерированного графа плюс отдельный поток атрибутов: **никаких мутаций структурного RNG/графа**.

Три правила, которые гарантируют AC (проверены по коду, п.3):
- R1. `build_*` в `schemes.py` и все существующие вызовы `rng` в `graph.py:48-151` — **не трогаем**.
- R2. Новые RNG-потребители живут ТОЛЬКО в отдельном потоке `default_rng(sha256(f"{seed}:attrs"))`.
- R3. Anchors/decoy/holdout — детерминированные вычисления без RNG.

---

## 2. Схема данных в памяти

### 2.1 `PatternInstance` (dataclass, новый модуль `anchors.py`)

Всякая посаженная структура (паттерн ИЛИ decoy p2p-блока). Строится по уже собранному `GeneratedGraph` — без единого вызова RNG.

| поле | тип | семантика |
|---|---|---|
| `instance_id` | `int` | глобальный 0-based, = порядку посадки (порядок `graph.nodes`) |
| `pattern_type` | `str` | `mixer\|peel_chain\|fanout\|hub_spoke\|wash\|p2p` (p2p — только decoy) |
| `node_ids` | `list[int]` | глобальные tx_id в порядке узлов схемы |
| `edge_ids` | `list[int]` | индексы в `graph.edges` (== номер строки edgelist / edge_attrs), в порядке builder-рёбер |
| `anchor_node_id` | `int` | `node_ids[0]` (builder-node 0: mixer_core / peel_hop_0 / scam / hub) |
| `anchor_role` | `str` | роль якоря из схемы |
| `max_anchor_distance` | `int` | эксцентриситет якоря внутри инстанса (наблюдаемый, см. п.9.1) |
| `temporal_window` | `(int,int)` | `(min_step, max_step)` узлов инстанса |
| `edge_attr_recipe` | `dict[int,str]` | `edge_id -> role`: `"mixer_in"\|"mixer_out"\|"peel"\|"wash"\|"star"\|"cycle"\|"bg"` |
| `decoy` | `bool` | `True` только для p2p-блоков, признанных decoy (п.7). Честные паттерны всегда `False` |
| `articulation_points` | `list[int]` | точки сочленения инстанса (Tarjan, детерминированно) — опц. поле, п.9.2 |

Правило привязки ребра к инстансу: **оба** конца — в `node_ids` этого инстанса. Это корректно по построению: `_register` соединяет только узлы своей схемы; bg-рёбра соединяют только bg-узлы (все bg-id > всех scheme-id, т.к. нумерация блочная). Пересечений нет, покрытие полное.

### 2.2 `GeneratedGraph` (расширение, graph.py)

Добавляются аннотационные поля (после сборки, без RNG):
- `instances: list[PatternInstance]`
- `instance_id_of: dict[int, int|None]` (tx_id -> instance_id; bg = None)

Сам процесс сборки `build_graph` — байт-в-байт прежний.

---

## 3. AC-сохранность: таблица RNG-потоков (проверено по коду)

**Структурный поток** = `np.random.default_rng(config.seed)` — `rng` инстанс в `build_graph` (graph.py:48). Его потребители:

| # | вызов | место | статус |
|---|---|---|---|
| 1 | `rng.shuffle(regular)` | graph.py:60 | неизм. |
| 2 | `rng.shuffle(novel)` | graph.py:61 | неизм. |
| 3 | `BUILDERS[name](rng, params)` → `_counts` (fan_in/fan_out/length/spokes/cycle_len) | graph.py:90, schemes.py:33-90 | неизм. (`build_*` не трогаем) |
| 4 | `stats.sample_step(rng, 1)` — birth | graph.py:92 | неизм. |
| 5 | `rng.random()` — drift gate | graph.py:96 | неизм. |
| 6 | `_offset_steps → rng.integers(0, OFFSET_RANGE)` | graph.py:31,69 | неизм. |
| 7 | `rng.integers(novel_step, STEP_MAX+1)` — novel birth | graph.py:106 | неизм. |
| 8 | `rng.shuffle(bg_pool)` | graph.py:130 | неизм. |
| 9 | `rng.integers(STEP_MIN, shutdown_step-1)` — bg illicit step | graph.py:138 | неизм. |
| 10 | `stats.sample_step(rng,1)` — bg step | graph.py:140 | неизм. |
| 11 | `rng.choice(bg_ids)` — src | graph.py:147 | неизм. |
| 12 | `rng.choice(bg_ids)` — dst | graph.py:148 | неизм. |

**Поток фич** = отдельный свежий экземпляр `default_rng(config.seed)`, создаваемый после `build_graph` (`__main__.py:29`, `tests:59`). Независим, порядок вызова не влияет на структурный.

**Атрибутный поток (НОВЫЙ)**:
```
attr_seed = int.from_bytes(sha256(f"{seed}:attrs").digest()[:8], "little")
rng_attr = np.random.default_rng(attr_seed)
```
Потребители: суммы edges (входы mixer, fee, сплит выходов, peel start/drip, wash, свободные суммы hub/fanout/p2p). Создаётся в emit-фазе, ПОСЛЕ всех структурных draws; не разделяет состояние ни с одним существующим инстансом → структурный поток не сдвигается. Детерминизм байт-в-байт сохраняется (см. п.10, тест).

**Детерминированные без RNG**: выбор якорей, BFS-окрестности, greedy-изоляция, decoy-детекция, Tarjan, holdout-разметка, timestamps.

Вывод: при дефолтном конфиге последовательность 1-12 — копия прежней, атрибуты/якоря/decoy не расширяют и не сужают граф → `10000 tx / 14092 edges` и сплит не меняются. Проверено живым прогоном: `Generated 10000 txs, 14092 edges`.

---

## 4. Edge-атрибуты

### 4.1 Конфиг `EdgeAttributeConfig`

```yaml
edge_attributes:
  amount:                      # диапазоны — качественные масштабы, draw = attr-RNG uniform
    mixer:       { fee_fraction: [0.005, 0.02], amount_per_in: [10000, 100000] }
    peel_chain:  { start_amount: [5000, 50000], drip: [0.50, 0.75] }
    wash:        { amount: [1000, 20000], balance_tolerance: 0.05 }
    hub_spoke:   { amount: [50, 5000] }
    fanout:      { amount: [50, 5000] }
    p2p:         { amount: [1, 1000] }
  timestamp_scale: 1200        # секунды: timestamp = 1200 * (max(step_u, step_v) - STEP_MIN)
```

Все суммы округляются до 2 знаков; "последний" ребро в каждом наборе поглощает остаток округления — инварианты точные.

### 4.2 CSV `elliptic_txs_edge_attributes.csv`

Колонки = формат реального Elliptic edge attributes:
```
txId1  int64    источник (как в edgelist)
txId2  int64    получатель
amount float64  >= 0, 2 знака
timestamp int64 = 1200 * (max(step_u, step_v) - STEP_MIN)   # детерминированно, без RNG
```
- Строк = `len(edgelist)` (== 14092 при дефолте), **та же нумерация строк**, что у рёбер → `edge_id` == номер строки == индекс в `graph.edges`.
- Порядок записей — порядок `graph.edges`; дополнительная сортировка не делается.

### 4.3 Инварианты (валидируются в п.8/п.10)

| паттерн | инвариант | конструкция |
|---|---|---|
| `mixer` | `sum(in) == sum(out) + fee`, fee = `round(total_in * f)`, f~U(fee_fraction) | 16-22 входов → core → выходы; out_budget = total_in - fee, сплит по взвешенным attr-RNG долям, последний выход поглощает копейки — сумма точная |
| `peel_chain` | `A0 >= A1 >= ... >= A_{L-1} >= 0`, строго убывает | `A_{i+1} = round(A_i * drip_i)`, `drip_i ~ U(drip)` < 1 |
| `wash` | per-node `sum(in) ≈ sum(out)` в пределах `balance_tolerance` | `A_i` заданы как `A_0 * prod r_j`, `r_j ~ U(1/(1+tol), 1+tol)`; баланс узла = `A_{i-1} - A_i = A_{i-1}(1-r_i)` |
| `hub_spoke` | свободный | spoke→hub и hub→spoke, без ограничений |
| `fanout` | свободный | scam→victims, без ограничений |
| `p2p` | свободный | малые суммы bg-рёбер (в т.ч. рёбра decoy-блоков) |

`balance_tolerance: 0.05` — документированный допуск, проверяется как `|sum(in)-sum(out)| <= tol * sum(in)`. Это НЕ точный ноль: wash-цикл — аннотация "≈0", а не бухгалтерия.

---

## 5. Anchors

### 5.1 Конфиг `AnchorsConfig`

```yaml
anchors:
  per_1000_nodes: 2
  min_anchor_distance: 3
  entity_types:
    mixer: wallet_mixer
    peel_chain: tumbler
    fanout: scam_propagator
    hub_spoke: exchange_hub
    wash: self_cycle
  k_hop_range: [1, 3]     # окрестности, для которых строятся gold-списки
```

### 5.2 Выбор якорей (детерминизм, без RNG)

1. Кандидаты = `anchor_node_id` каждого не-p2p инстанса, в порядке `instance_id` (порядок посадки — уже детерминирован структурным потоком).
2. Бюджет: `target = per_1000_nodes * (n_txs / 1000)` (при n_txs=10000, per_1000=2 → 20).
3. **Greedy-изоляция**: идём по кандидатам (возрастание instance_id); держим якорь, если BFS-расстояние в `graph.edges` до всех ранее удержанных якорей `>= min_anchor_distance` (BFS срезан на глубине `min_anchor_distance`; соседи перебираются по возрастанию tx_id). Иначе — отбрасываем кандидата, но инстанс остаётся в `instance_ground_truth` (просто не якорный).
4. Если удержанных якорей меньше `target` — выдать сколько есть, **не добирать** (никогда не выдумывать якоря). Никакого RNG.

### 5.3 Якорные окрестности

Для каждого удержанного якоря: `k_hop_neighborhoods[k]` = узлы BFS-хопа k внутри его инстанса (для k=1..`max_anchor_distance`). `k_hop_range` задаёт, до какого k зафиксировать массивы в манифесте (глубже — не выводим).

---

## 6. Manifest — точная JSON-схема

Заголовок manifest не меняется, ключи добавляются аддитивно. Структура:

```
seed: int
config: {...}                     # to_dict + НОВЫЕ блоки edge_attributes/anchors/background/holdout
sha256: {features, classes, edgelist, edge_attributes: NEW}
ground_truth: [per-node]          # буд. ключи txId/scheme/role/class + NEW instance_id (int|None; bg=null)
instance_ground_truth: [ ... ]    # NEW
edge_attribute_ground_truth: [ ... ]  # NEW
anchor_registry: { ... }          # NEW
retrieval_specs: { ... }          # NEW
decoys: [ ... ]                   # NEW
holdout: { windows_active: bool, entries: [...] }  # NEW
```

### Полный пример (реальные числа с прогона seed 42, instance 0 и 1 — подлинные; amounts — иллюстративные, сам атрибутный семплинг новый):

```json
{
  "seed": 42,
  "config": {
    "seed": 42, "n_txs": 10000, "labeled_ratio": 0.23,
    "illicit_ratio_in_labeled": 0.10, "p2p_edges_per_tx": 1.5,
    "stats_dir": "data/elliptic_stats", "schemes": {},
    "drift": { "enabled": false, "shutdown_step": 40,
               "shutdown_rate_multiplier": 0.0, "novel_step": 45, "novel_scheme": "wash" },
    "edge_attributes": { "amount": { "mixer": {"fee_fraction": [0.005, 0.02], "amount_per_in": [10000, 100000]},
                                     "peel_chain": {"start_amount": [5000, 50000], "drip": [0.5, 0.75]},
                                     "wash": {"amount": [1000, 20000], "balance_tolerance": 0.05},
                                     "hub_spoke": {"amount": [50, 5000]},
                                     "fanout": {"amount": [50, 5000]},
                                     "p2p": {"amount": [1, 1000]} },
                         "timestamp_scale": 1200 },
    "anchors": { "per_1000_nodes": 2, "min_anchor_distance": 3,
                 "entity_types": {"mixer":"wallet_mixer","peel_chain":"tumbler",
                                  "fanout":"scam_propagator","hub_spoke":"exchange_hub","wash":"self_cycle"},
                 "k_hop_range": [1, 3] },
    "background": { "decoy_detection": true, "fan_in_threshold": 3, "cycle_max_depth": 4 },
    "holdout": { "entries": [] }
  },
  "sha256": {
    "features": "…", "classes": "…", "edgelist": "…",
    "edge_attributes": "…"                    // новый sha256
  },
  "ground_truth": [
    { "txId": 0,   "scheme": "mixer", "role": "mixer_core",  "class": "illicit", "instance_id": 0 },
    { "txId": 1,   "scheme": "mixer", "role": "mixer_entry", "class": "licit",   "instance_id": 0 },
    { "txId": 23,  "scheme": "fanout", "role": "scam",       "class": "illicit", "instance_id": 1 },
    { "txId": 24,  "scheme": "fanout", "role": "victim",     "class": "licit",   "instance_id": 1 },
    { "txId": 1645, "scheme": "p2p",  "role": "background",  "class": "unknown", "instance_id": null }
  ],
  "instance_ground_truth": [
    {
      "instance_id": 0, "pattern_type": "mixer",
      "anchor_node_id": 0, "anchor_role": "mixer_core",
      "entity_type": "wallet_mixer", "max_anchor_distance": 2,
      "temporal_window": [11, 13],
      "node_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
      "edge_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
      "articulation_points": [0],
      "anchored": true,
      "holdout": false, "train_excluded": false
    },
    {
      "instance_id": 1, "pattern_type": "fanout",
      "anchor_node_id": 23, "anchor_role": "scam",
      "entity_type": "scam_propagator", "max_anchor_distance": 1,
      "temporal_window": [34, 36],
      "node_ids": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],
      "edge_ids": [22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                   42, 43, 44, 45],
      "articulation_points": [23],
      "anchored": false,          // пример: не прошел greedy-изоляцию (дистанция < min_anchor_distance до якоря 0)
      "holdout": false, "train_excluded": false
    }
  ],
  "edge_attribute_ground_truth": [
    { "edge_id": 0,  "txId1": 1,  "txId2": 0,  "pattern_type": "mixer", "instance_id": 0, "role": "mixer_in",  "amount": 43210.50 },
    { "edge_id": 15, "txId1": 16, "txId2": 0,  "pattern_type": "mixer", "instance_id": 0, "role": "mixer_in",  "amount": 21000.00 },
    { "edge_id": 16, "txId1": 0,  "txId2": 17, "pattern_type": "mixer", "instance_id": 0, "role": "mixer_out", "amount": 62100.00 },
    { "edge_id": 22, "txId1": 23, "txId2": 24, "pattern_type": "fanout","instance_id": 1, "role": "star",       "amount": 99.90 }
  ],
  "anchor_registry": {
    "per_1000_nodes": 2, "min_anchor_distance": 3,
    "anchors": [
      { "instance_id": 0, "anchor_node_id": 0, "anchor_role": "mixer_core",
        "entity_type": "wallet_mixer", "pattern_type": "mixer", "max_anchor_distance": 2,
        "k_hop_neighborhoods": { "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                 "2": [], "3": [] } }
    ]
  },
  "retrieval_specs": {
    "node_vector_dim": 167,
    "vectors_file": "elliptic_txs_features.csv",
    "node_id_column": "txId",
    "node_vector_lookup": "row of vectors_file where txId column == node id",
    "anchor_key": "anchor_node_id",
    "entity_type_key": "entity_type",
    "neighborhood_field": "k_hop_neighborhoods",
    "default_k": 10
  },
  "decoys": [
    { "nodes": [5123, 5188, 5641, 6012], "edge_ids": [9977, 10012, 10089],
      "motif": "fan_in", "pattern_type": "p2p" },
    { "nodes": [7001, 7002, 7003], "edge_ids": [11901, 11902, 11903],
      "motif": "cycle", "pattern_type": "p2p" }
  ],
  "holdout": { "windows_active": false, "entries": [] }
}
```

Правила схемы:
- `ground_truth[*].instance_id` — новый ключ; `scheme/role/class` и их тайп-контракт (строка, `1/2/unknown` в классе) не тронуты → старый парсинг совместим.
- `edge_id` в `edge_attribute_ground_truth` = номер строки edgelist/edge_attrs; `amount` дублирует CSV (удобство бенчмаркера, источник истины — CSV + sha256).
- `k_hop_neighborhoods`: только k ≤ `min(max_anchor_distance, max(k_hop_range))`; пустые хопы не пишем, если хоп пуст — пишем `[]` (как в примере для mixer, диаметр 2).

---

## 7. Background cleaner + decoy (детерминизм, без RNG)

Детекция только среди **p2p-рёбер** (оба конца `scheme=="p2p"`):

- **fan_in**: bg-узел с `in_degree >= fan_in_threshold` (3) от различных bg-источников. Порядок обхода — возрастание tx_id. В decoy → узел-цель + её источники + edge_ids.
- **cycle**: DFS по bg-подграфу (adjacency строится отсортированной по tx_id), глубина ≤ 4; обнаружение back-edge на глубине ≤ 4 → цикл (записывается минимальный возв-цикл по отсортированным узлам). В decoy → узлы цикла + edge_ids.

Аннотация чистая: **граф не мутируется**, блоки НЕ удаляются и НЕ переподписываются. Вывод — в `manifest.decoys`. Это «камуфляж-кандидаты»: похожи на схемы, но не настоящие инстансы; неизвестно-класс, p2p. Для бенчмарка — базовый уровень верных срабатываний (decoy должен оставаться ниже порога).

---

## 8. Holdout (аннотация, обобщение drift-логики)

```yaml
holdout:
  entries: []      # каждый элемент { pattern, step_min, step_max, train_excluded }
```

- `drift` (существующий) управляет **эмиссией** wash в поздние шаги — без изменений. wash остаётся drift-novel.
- `holdout.entries` управляет **разметкой**: инстанс попадает в окно, если `pattern` совпал и `temporal_window ∩ [step_min, step_max] ≠ ∅`. Тогда в `instance_ground_truth[instance]`: `holdout: true`, `train_excluded: <значение>`. Никаких мутаций графа, эмиссии, RNG.
- Пустой `entries` → `windows_active: false`, ни один инстанс не помечен → поведение файлов идентично текущему (manifest отличается только аддитивными ключами, которые детерминированы между прогонами — AC байт-детерминизма проходит, см. п.3/п.10).
- **RNG-безопасность нового пула**: на этой итерации новый пул не создаётся (аннотация детерминированна). Если позже понадобится «пересадить» паттерн в окно активации (мутация, как drift.novel), сэмплинг birth ОБЯЗАН идти из `default_rng(sha256(f"{seed}:holdout"))`, никогда из структурного `rng` — это единственный способ не сдвинуть AC.

---

## 9. Риски / гэпы (принято / отклонено, "ленивый сеньор")

| пункт spec | решение | обоснование |
|---|---|---|
| Якорь-расстояние фикp. **2-4** | **ОТКЛОНЕНО как гарантия; принято как наблюдаемое** | Текущие билдеры дают эксцентриситет якоря: `fanout`/`hub_spoke` = 1, `mixer` = 2, `peel_chain` = L-1 (до 9), `wash` = ⌊L/2⌋. Гарантировать [2,4] = менять `build_*` = сдвиг структурного RNG = смерть AC. Вместо этого: `max_anchor_distance` — наблюдаемое поле, `k_hop_range [1,3]` — аннотация запросов. Гомоморфизм "якорь → соседи ≤2" истинен для mixer/fanout/hub_spoke — покрывает спец-кейсы бенчмарка. `# ponytail: дальше окрестностей 2-hносопчета без новых билдеров не идем; путь апгрейда — инстанс с 2+ хопами в глубину только если Spillety мерит recall@k для k>2 и захочет явно.` |
| **bottleneck** | **ОТКЛОНЕНО как посаженный мотif; принято как производная аннотация** | Новый мотifbottleneck = новый builder = сдвиг RNG/структуры, и схемы в спецификации v1 его не содержат. Но узкие места уже существуют структурно: core mixer'а и средние узлы peel_chain. Выдаём их детерминированно через Tarjan (O(V+E), без RNG) в `instance_ground_truth[*].articulation_points`. Этого достаточно, пока Spillety не попросит явный компонент-бутылочку. |
| Holdout как **мутация графа** (пересадка паттернов) | **ОТКЛОНЕНО в пользу аннотации** | Мутация = новый RNG-потребитель/пул = риск сдвига структурного потока. Аннотация решает задачу «train_excluded» и «окно активации» в бенчмарке без единого random draw. Правило безопасного апгрейда зафиксировано (п.8). |
| Decoy-детекция **режет/переписывает граф** | **ОТКЛОНЕНО: только флаг в manifest** | Decoy уже в графе; удаление рёбер/узлов сломало бы 14092 edges AC. Блокирование/фильтр — сторона бенчмаркера. |
| `edge_attributes.enabled` флаг | **ОТКЛОНЕНО: файл всегда эмитируется** | Файл не меняет числа AC (граф тот же), детерминирован, а флаг — мёртвая ветка. YAGNI. |

Прочие гэпы:
- **temporal split 5856:2050:2094** — это границы train/val/test по steps (рассчитываются в Spillety по `time_step`), генератор их не знает; AC-проверка сплита остаётся в тестах/README, а не в manifest (узел-вектор имеет только features/step).
- **Корреляции сумм и объёма** не моделируются (суммы на 2 знака, без модели потока). Осознанное упрощение уровня "marginal-only".
- **wash при drift.off отсутствует → decoy-детекция и якоря для wash не тестируются на дефолте**; покрытие — параметризованные тесты с `drift.enabled: true` (как `test_drift_acceptance`, сиды 0..N).
- **Один edge_id может фигурировать и в decoy, и в choke-графе** (bg-ребро внутри decoy-блока) — в манифесте разрешено: роли подграфов могут пересекаться по рёбрам p2p.

---

## 10. Файл → изменения

| файл | изменение |
|---|---|
| `src/kyt_engine/synth/anchors.py` | **НОВЫЙ**: `build_instance_index(graph,cfg)`, `build_edge_attributes(graph,instances,cfg,rng_attr)`, `select_anchors(instances,graph,cfg)`, `detect_decoys(graph,cfg)`, `apply_holdout(instances,cfg)`; `ATTR_SEED = sha256(f"{seed}:attrs")`, `HOLDOUT_SEED` (не используется, договорённость); `ENTITY_TYPES` внешний маппинг. |
| `graph.py` | после сборки — детерминированная аннотация `instances` + `instance_id_of` (на тех же циклах, 0 RNG-вызовов). |
| `emit.py` | запись `elliptic_txs_edge_attributes.csv` (4 колонки, порядок `graph.edges`); manifest: новые ключи (`sha256.edge_attributes`, `instance_id` в per-node GT, `instance_ground_truth`, `edge_attribute_ground_truth`, `anchor_registry`, `retrieval_specs`, `decoys`, `holdout`); вызов anchors-функций. |
| `config.py` | dataclassы `EdgeAttributeConfig`, `AnchorsConfig`, `BackgroundConfig`, `HoldoutConfig` (default `entries=[]`, decoy on), включение в `to_dict`/`from_yaml`. |
| `validate.py` | **опциональный** `validate_edge_attributes(root, manifest)` — вызывается отдельно, `validate_dataset`/`load_elliptic` signature без изменений (контракт loader не расширяем). Проверки: число строк == edgelist, колонки, amount >= 0, invariants п.4.3 по `edge_attribute_ground_truth`. |
| `configs/generator.yaml` | новые блоки `edge_attributes/anchors/background/holdout` с дефолтами + комментарий «не трогаем RNG-порядок». |
| `tests/test_synth.py` | `OUT_FILES` += edge_attributes; новые: (а) детерминизм всех 5 файлов + manifest, (б) инварианты 4.3 (mixer/peel/wash через drift-on конфиг), (в) якоря: `anchored: true` якоря попарно на дистанции ≥ min_anchor_distance; (г) decoy: `motif` валиден, узлы ⊆ p2p, повторный прогон идентичен; (д) holdout: пустой entries → `windows_active:false`; с записями → помечены инстансы попадающие в окно. |
| `README.md` | новый формат: 4 CSV + расширенный manifest; описание якорей/retrieval/decoy/holdout. |

## 11. Верификация

- `make test` / `make lint` — зелёные.
- Живой прогон дефолта: 10000 tx / 14092 edges / сплит-гистограмма по steps идентична до-рефакторинговой.
- Два прогона одного seed → byte-identical во всех 5 CSV за 3 CSV + manifest (включая новые ключи).
- `validate` (контракт loader) — нетто без изменений; edge-валидация — отдельная команда.