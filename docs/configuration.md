# Конфигурация

## Canonical behavior config

Основной файл запуска: `configs/generator_behavior.yaml`. В behavior-only версии в нём нельзя включить другую генерацию признаков или другой graph. Значения `graph.mode: behavior` и `features.mode: semantic` оставлены как явные маркеры контракта.

Пример canonical-конфигурации:

```yaml
seed: 72
n_txs: 10000
stats_dir: data/volume
out_dir: data/synthetic/behavior_run

graph:
  mode: behavior
features:
  mode: semantic

behavior:
  contract_version: behavior.p0.v1
  profile_quotas:
    ordinary_wallet: {entity_count: 120, tx_quota: 4000}
    exchange_hub: {entity_count: 12, tx_quota: 1200}
    miner_payout: {entity_count: 20, tx_quota: 800}
    wallet_provider: {entity_count: 30, tx_quota: 1000}
    individual: {entity_count: 180, tx_quota: 1800}
    mixer: {entity_count: 8, tx_quota: 600}
    wash_round_trip: {entity_count: 6, tx_quota: 600}
  chains:
    bitcoin: {enabled: true}
    ethereum: {enabled: true}
    tron: {enabled: true}
  drift:
    phases:
      - phase_id: stable
        step_min: 1
        step_max: 39
        illicit_keep_probability: 1.0
        novel_profile_ids: []
      - phase_id: shutdown
        step_min: 40
        step_max: 44
        illicit_keep_probability: 0.0
        novel_profile_ids: []
      - phase_id: novel
        step_min: 45
        step_max: 49
        illicit_keep_probability: 0.0
        novel_profile_ids: [wash_round_trip]

edge_attributes:
  timestamp_scale: 1200

anchors:
  per_1000_nodes: 2
  min_anchor_distance: 3
  k_hop_range: [1, 3]
  prefer_central_anchors: true

holdout:
  entries: []

gnn:
  backend: auto
  hidden: 64
  layers: 2
  dropout: 0.25
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 300
  patience: 30
  threads: 1
  k: 100
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Все optional sections можно не указывать, если принять defaults. В canonical file они записаны явно, чтобы manifest и report отражали effective configuration без догадок.

## Основные поля

### `seed` и `n_txs`

`seed: 72` является canonical seed для standalone behavior run. Он должен быть явно записан в config, потому что manifest и все derived IDs зависят от него.

`n_txs` задаёт точное число целевых transactions. Сумма `tx_quota` всех profiles должна равняться `n_txs`. Проверка выполняется до создания RNG stream. Если подходящих step slots не хватает, run завершается ошибкой, а не добивается quota случайными nodes.

### `stats_dir`

`stats_dir` указывает каталог с единственным временным входом:

```text
data/volume/volume.npy
```

Массив имеет форму `(49,)`, содержит неотрицательные целые веса и используется методом largest-remainder для распределения slots по шагам. При нулевой сумме scheduler использует равномерный fallback, но это не оценка распределения Elliptic.

Эмпирические distributions признаков в этом проекте не используются. В config нет отдельных путей к таким артефактам.

### `out_dir`

В `out_dir` writer создаёт четыре CSV и `manifest.json`. Путь не должен находиться внутри `data/raw`. Для воспроизводимого bundle не смешивайте два run в одном каталоге без явного удаления старых файлов.

## `behavior.profile_quotas`

Каждый profile получает две величины:

- `entity_count`: количество сущностей этого типа;
- `tx_quota`: количество целевых transactions, которые должен создать profile.

Значения из canonical config дают `10000` transactions. Они являются budgets симуляции, а не измеренными долями Elliptic. Если изменить `n_txs`, нужно изменить quota так, чтобы сумма оставалась точной.

Положительная `tx_quota` требует положительный `entity_count`. Неизвестный profile ID, отсутствие action policy или нулевой counterparty budget приводят к fail-closed validation.

## Chains

Поведенческий registry содержит `bitcoin`, `ethereum` и `tron`. В config должен быть указан весь набор chain keys, а хотя бы одна chain должна быть enabled. Bridge action требует минимум две enabled chains, иначе cross-chain policy невозможна.

Chain selection выполняется из profile weights отдельным population RNG. Это не означает, что суммы BTC, ETH и TRX имеют общую экономическую единицу. В CSV используется normalized amount, а native asset остаётся typed metadata.

## Drift phases

Phases должны:

- идти в возрастающем порядке;
- начинаться с шага `1`;
- заканчиваться ровно на шаге `49`;
- быть contiguous и не пересекаться;
- иметь уникальные `phase_id`;
- использовать вероятность в диапазоне `0..1`;
- ссылаться только на зарегистрированные profiles.

В `shutdown` с `illicit_keep_probability: 0.0` обычные illicit profiles не проходят eligibility. В `novel` profile из `novel_profile_ids` должен иметь policy `novel_illicit`. Manifest сохраняет configured schedule и realized counts.

## Edge attributes и anchors

`edge_attributes.timestamp_scale` задаёт масштаб вычисляемого timestamp. Он не задаёт units реальной сети и не должен интерпретироваться как wall-clock time.

Anchor settings влияют только на retrieval metadata:

- `per_1000_nodes` задаёт целевое число anchors;
- `min_anchor_distance` разделяет слишком близкие anchors;
- `k_hop_range` задаёт записываемые neighborhood depths;
- `prefer_central_anchors` входит в effective config; в behavior projection используется канонический центральный anchor.

Anchor selection не меняет transaction graph и не добавляет transaction IDs.

`holdout.entries` помечает instances, чьи temporal windows пересекают указанный диапазон. Это annotation, а не удаление graph nodes или edges.

## GNN configuration

Секция `gnn` используется только strict downstream.

| Поле | Canonical значение | Смысл |
|---|---:|---|
| `backend` | `auto` | full PyG при наличии, иначе явный SGC diagnostic |
| `hidden` | `64` | размер hidden representation |
| `layers` | `2` | число слоёв full model |
| `dropout` | `0.25` | dropout |
| `learning_rate` | `0.001` | learning rate |
| `weight_decay` | `0.0001` | weight decay |
| `epochs` | `300` | максимум эпох |
| `patience` | `30` | early stopping patience |
| `threads` | `1` | CPU threads |
| `k` | `100` | ranking budget для P@K и R@K |
| `seeds` | `0..9` | frozen roster для real acceptance |

`hidden` должен быть чётным, `layers` может быть `2` или `3`, а все dimensions, epochs, patience, threads и `k` должны быть положительными. Seeds не должны повторяться.

CLI overrides для `validate-real` имеют приоритет только при явном флаге. `backend`, `k` и `seeds` можно переопределить на запуске. `threads` и model hyperparameters берутся из YAML. В report публикуются effective values и источник каждого override.

## Seed namespaces

Derivation выполняется по формуле:

```text
digest = sha256(f"{seed}:{namespace}")
state = little-endian uint64 из digest[0:8]
```

Namespace list фиксирован:

| Namespace | Ответственность |
|---|---|
| `behavior:population` | entities и chain assignment |
| `behavior:schedule` | step slots и profile assignment |
| `behavior:transition` | action, actor и counterparty |
| `behavior:amount` | amount policy |
| `behavior:drift` | eligibility и phase gate |

Все streams используют PCG64. Общий generator state, `hash()` и filesystem order не используются. Seed `72` в config и seed roster `0..9` в acceptance имеют разные роли и не должны смешиваться.

## Строгая проверка YAML

Parser отклоняет:

- неизвестные top-level и nested fields;
- duplicate keys;
- `bool` вместо integer;
- non-finite floats;
- отрицательные quotas и dimensions;
- неверные profile, chain или action IDs;
- пустые или неполные drift phases;
- `graph.mode`, отличный от `behavior`;
- `features.mode`, отличный от `semantic`.

Проверка выполняется до RNG и до записи output. Это позволяет отличить неверную конфигурацию от неуспешного downstream experiment.
