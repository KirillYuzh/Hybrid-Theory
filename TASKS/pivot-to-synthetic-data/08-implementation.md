# 08-implementation.md — Реализация retrieval-бенчмарка (задача «разбей на задачи и на subagents и реализуй»)

Статус: **завершено**, 0 🔴 Critical, 0 ai-slops, AC выполнены (включая старые AC-числа).

## Задача

«Spillety-Aligned Synthetic Retrieval Bench»: разбить на задачи и subagents и реализовать поверх существующего генератора слой контролируемого retrieval-бенчмарка — инстансы (посаженные подграфы), якоря (корни запросов), edge-атрибуты с проверяемыми инвариантами, pattern-level holdout, decoy-разметку фона.

## Как выполнено (под-задачи)

1. **Design-subagent** → артефакт `07-retrieval-bench-design.md`: контракты, RNG-таблица, gap-анализ спекуляции. Главные решения:
   - AC-preservation: **R1** — не трогать `build_*`/структурный `default_rng(seed)`; **R2** — новые RNG-потребители только в `default_rng(int.from_bytes(sha256(f"{seed}:attrs")[:8], "little"))`; **R3** — якоря/decoy/holdout/BFS детерминированы без RNG;
   - edge-атрибуты упрощены до `amount` + `timestamp` (стиль реального Elliptic); bottleneck → точки сочленения (Tarjan); decoy — аннотация без мутации графа; `edge_attributes.enabled` — YAGNI, всегда пишем CSV;
   - holdout обобщает drift **аддитивно** (drift никак не затронут, holdout — только аннотация).
2. **M1 (инстансы)** — `PatternInstance`; `graph.py`: `SchemeRun` + запись границ в `_register` (0 RNG), `local_to_global`/`node_id_start` починены.
3. **M2 (якоря)** — `select_anchors` greedy по порядку инстансов, изоляция ≥ `min_anchor_distance` по undirected BFS в полном графе; фикс `_bfs_distance(...) or 0` → `_too_close` (дистанция «дальше капа» не отвергает кандидата); `k_hop_neighborhoods` внутри инстанса.
4. **M4 (манифест-контракт)** — `anchor_registry`, `retrieval_specs` (descriptor), `instance_ground_truth`, `edge_attribute_ground_truth`, `decoys`, `holdout`; `instance_id` в per-tx `ground_truth`.
5. **M6 (edge-attributes)** — CSV `elliptic_txs_edge_attributes.csv` (одна строка на ребро edgelist), суммы в центах: mixer `sum(in)=sum(out)+fee` (последний out поглощает остаток — равенство точное), peel c `drip` и флором, wash с клиппингом кумулятивного дрейфа и замыкающим ребром (каждый узел балансируется), star/p2p свободные.
6. **M5 (holdout)** — `apply_holdout` аннотирует инстансы по пересечению `temporal_window` с окнами entries; неполные записи пропускаются.
7. **M3 (background/decoy)** — `detect_decoys`: fan_in ≥ 3 и cycle ≤ 4 в p2p-фоне; детерминированно (сортировки), без RNG; выключается конфигом.

## Что изменено

- **`src/kyt_engine/synth/anchors.py`** (новый, ~430 строк) — инстансы, edge-атрибуты, `attr_seed`/`attr_rng`, якоря, K-hop, decoy, holdout, консолидированный `_bfs_layers`.
- **`graph.py`** — `SchemeRun` (границы инстансов, без RNG), фиксы `local_to_global`.
- **`config.py`** — `EdgeAttributeConfig`, `AnchorsConfig`, `BackgroundConfig`, `HoldoutConfig`; `from_yaml`: amount/entity_types мержатся с дефолтами (частичный YAML не падает); deepcopy-дефолты.
- **`emit.py`** — 4-й CSV; manifest-блоки retrieval-контракта; sha256 включает edge_attributes; `retrieval_specs.node_vector_lookup` = первая без-header колонка, `default_k = max(k_hop_range)`.
- **`validate.py`** — `validate_edge_attributes` (структура; инварианты — в тестах, задокументировано). **`__main__.py`** — вызов в CLI.
- **`configs/generator.yaml`** — секции `edge_attributes`/`anchors`/`background`/`holdout`; комментарий про peel-флор; trailing newline.
- **`tests/test_synth.py`** — 5 новых тестов; константы читаются из `cfg` (не зашиты); mirror-BFS помечен `# ponytail:`.

## Ревью

`artifacts/subagents/review-retrieval-bench.md`: **0 Critical**, 11 Suggestions + 7 ai-slop-пунктов. Закрыты все ai-slops (мёртвый `PatternInstance.decoy` и guards, неиспользуемый `config` в `_instance_entry`, 7 копий dict-литерала → helper `entry`, `list[dict|None]`, docstring-пересказы, trailing newline) и Suggestions: S1 (peel-флор документирован в конфиге + тест на монотонность), S2 (cycle не дублирует замыкающий узел, `len(nodes)==len(edges)`), S4 (merge amount/entity_types), S5 (пустой `k_hop_range` не крашит), S6 (retrieval_specs под headerless CSV), S7 (тест читает конфиг), S8 (пропуск неполных holdout-записей), S10 (deepcopy), S11 (консолидированный `_bfs_layers`). S9 (`out_dir` в `to_dict`) — отклонён: ломает byte-детерминизм manifest между разными out_dir.

## Итоговая верификация

| AC | Значение |
|---|---|
| generate → 10 000 tx / 14 092 edges (не изменились) | ✅ |
| temporal_split 5856 / 2050 / 2094 (не изменились) | ✅ |
| детерминизм byte-identical 5 файлов (2 CLI-прогона, diff пуст) | ✅ |
| дрифт: wash 45–49, non-wash illicit < 40 (сиды 0..19) | ✅ |
| якоря 20/20 при `per_1000_nodes: 2`, изоляция ≥ 3 (190 пар BFS) | ✅ |
| k-hop ⊆ инстанса | ✅ |
| decoys 1599 (1598 fan_in + 1 cycle), cycle: уникальные узлы, edges==nodes | ✅ |
| инварианты: mixer fee ∈ [0.004,0.021], peel монотонен, wash-баланс в tolerance | ✅ |
| `make test` 16 passed, `make lint` clean | ✅ |

## Документация

README («Retrieval-бенчмарк»), `docs/how-it-works.md` (новая §11 + §12..§18 перенумерованы), `.agents/skills/generator/SKILL.md`, `K-BRAIN/TODO.md`.