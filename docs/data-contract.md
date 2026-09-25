# Контракт данных

## Состав набора файлов

Один behavior run записывается в каталог `out_dir` и содержит ровно пять файлов:

1. `elliptic_txs_features.csv`;
2. `elliptic_txs_classes.csv`;
3. `elliptic_txs_edgelist.csv`;
4. `elliptic_txs_edge_attributes.csv`;
5. `manifest.json`.

Четыре CSV сохраняют раскладку Elliptic, чтобы существующий loader мог прочитать набор. Поведенческая provenance хранится в `manifest.json` и не добавляется в CSV.

## `elliptic_txs_features.csv`

Файл записывается без заголовка и без индекса. Каждая строка содержит 167 колонок:

```text
txId,time_step,feat_2,feat_3,...,feat_166
```

Из них:

- `txId` и `time_step` занимают первые две позиции;
- `feat_2..feat_166` содержат 165 признаков;
- `txId` и `time_step` являются локальными IDs и целыми числами;
- все значения признаков должны быть конечными.

`txId` начинается с нуля и идёт без пропусков в порядке строк. Время находится в диапазоне `1..49`. Генератор не обещает фиксированное число transactions, если пользователь изменил `n_txs`.

### Семантические колонки

Раскладка записывается в `manifest.feature_semantics`. В текущем behavior emitter:

- `feat_2..feat_25` содержат 24 признака топологии, сумм и времени;
- `feat_26..feat_30` содержат chain one-hot и cross-chain degree признаки;
- `feat_31..feat_166` являются детерминированными производными преобразованиями semantic block.

Производные колонки нужны для совместимой ширины Elliptic, но не имеют отдельной интерпретации реального Elliptic feature. Потребитель должен читать `feature_semantics`, а не угадывать смысл по номеру колонки.

## `elliptic_txs_classes.csv`

Файл имеет заголовок:

```csv
txId,class
```

Возможные значения `class`:

- `1`;
- `2`;
- `unknown`.

Порядок строк совпадает с порядком `elliptic_txs_features.csv`. Каждый `txId` встречается один раз. В behavior-only run class берётся из target action policy. Entity classes не заменяют эту transaction-level строку.

Значение `unknown` остаётся частью формата, потому что его используют реальные Elliptic partitions. В downstream оно не является ни положительным, ни отрицательным классом.

## `elliptic_txs_edgelist.csv`

Файл имеет заголовок:

```csv
txId1,txId2
```

Это направленный edgelist. Обе колонки содержат локальные transaction IDs из features:

- `txId1` соответствует источнику;
- `txId2` соответствует цели;
- endpoints должны существовать в features;
- behavior bundle не содержит self-loop;
- duplicate directed pair отклоняется validator;
- `edge_id` равен индексу строки, начиная с нуля.

`edge_id` не является отдельной колонкой CSV. Он используется как row ID в manifest и при ссылке на event, instance и chain record.

## `elliptic_txs_edge_attributes.csv`

Файл имеет заголовок:

```csv
txId1,txId2,amount,timestamp
```

Число строк и порядок `txId1,txId2` должны точно совпадать с edgelist. Для каждой строки:

- `amount` конечен и не отрицателен;
- `timestamp` целый и не отрицательный;
- amount является нормализованной синтетической величиной;
- timestamp вычисляется из временных шагов источника и цели;
- значения не являются native-unit amount или wall-clock timestamp.

Native asset, chain и bridge information находятся в manifest. Суммы разных chains не складываются в CSV как одна экономическая величина.

## `manifest.json`

Manifest является канонической provenance-записью для этого bundle. Его основные части:

| Ключ | Содержание |
|---|---|
| `seed` | canonical behavior seed данного run |
| `config` | effective behavior configuration |
| `sha256` | хеши четырёх CSV |
| `ground_truth` | одна запись на каждую transaction с `event_id`, `entity_ids` и `chain_id` |
| `instance_ground_truth` | node, edge, event, entity и temporal membership для instances |
| `edge_attribute_ground_truth` | row-indexed edge references и action metadata |
| `anchor_registry` | выбранные anchors и k-hop neighborhoods |
| `retrieval_specs` | соответствие между CSV rows, vectors и anchor IDs |
| `feature_semantics` | layout и происхождение 165 feature columns |
| `decoys` | пустой behavior-only список |
| `holdout` | configured windows и instance annotations |
| `behavior` | полный behavior provenance block |

Behavior block включает как минимум:

- `contract_version`;
- `behavioral_profiles` с registry version, registry hash и node references;
- `entities`;
- `events`;
- `instances`;
- `drift_schedule` с configured и realized частями;
- `chain_metadata` с chain registry, node, edge и bridge records;
- `sanctions_anchor_map` в режиме `provenance_only`;
- `calibration` с явным `unknown_policy` и status; в текущем run `status=not_evaluated`, а score имеет `uncalibrated_model_score`;
- `rng_contract` с `PCG64` и namespace list.

## Связи между частями

Проверяемая цепочка ссылок выглядит так:

```text
features.txId
  -> classes.txId
  -> ground_truth.txId
  -> behavior.events.target_tx_id
  -> behavior.entities
  -> behavior.instances
  -> edgelist row
  -> edge_attributes row
  -> chain_metadata edge record
```

Один event создаёт ровно один target node. Source IDs из event должны существовать и быть причинно предыдущими. Instance membership задаётся явными ID, а не предполагается по порядку строк.

Entity IDs, event IDs, action IDs, instance IDs и native IDs не являются колонками Elliptic CSV. Они остаются только в manifest, поэтому CSV сам по себе нельзя использовать для полного анализа поведения.

## Manifest и целостность

`sha256` вычисляется по байтам каждого CSV. После ручного изменения CSV hash перестаёт совпадать, и validation завершается ошибкой. Проверка также сравнивает row alignment, semantic matrix, class coverage, edge endpoints, event causality, entity references, instance membership, chain records и drift realization.

В manifest нет реальных sanctions addresses и нет утверждения, что synthetic native ID принадлежит реальной сети. `sanctions_anchor_map` намеренно остаётся пустым provenance-only блоком.

## Acceptance генератора

Поведенческий bundle считается пригодным для следующего этапа, если выполнены все локальные условия:

- config прошёл строгую проверку;
- число transactions равно `n_txs`;
- `txId` непрерывны, а CSV classes идут в том же порядке;
- каждый event покрыт ровно одной target transaction;
- source references существуют и не ссылаются в будущее;
- instance, entity, edge и chain records покрыты manifest;
- semantic matrix и edge attributes восстанавливаются из графа;
- SHA-256 hashes совпадают с байтами CSV;
- один и тот же config, volume и seed дают тот же bundle.

Это локальная проверка формата и provenance. Она не является оценкой качества на реальных данных.

## Проверка перед использованием

Перед передачей bundle в другой процесс выполните:

```bash
python -m kyt_engine.synth validate \
  --dir data/synthetic/behavior_run
```

Успешная проверка означает согласованность файлов и provenance. Она не означает, что synthetic distribution совпадает с Elliptic или что модель достигнет acceptance target.
