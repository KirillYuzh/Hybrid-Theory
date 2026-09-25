# Strict-inductive downstream

## Назначение

Downstream проверяет, можно ли использовать модель, обученную на synthetic behavior bundle, для transfer на реальные Elliptic transactions. Это не проверка качества всех 165 semantic columns. Для transfer используется отдельное общее пространство структурных признаков, которое можно одинаково посчитать по synthetic и raw graph.

Протокол версионирован и находится в `src/kyt_engine/synth/gnn_downstream.py`. Его результат записывается в JSON report и не меняет raw data.

## Временные partitions

Все transactions имеют `time_step` от `1` до `49`. Разбиение фиксировано:

| Partition | Шаги | Назначение |
|---|---|---|
| `train` | `1..30` | обучение и подбор preprocessing на synthetic source |
| `validation` | `31..40` | early stopping и выбор threshold |
| `test` | `41..49` | финальное scoring на synthetic и target |

Перед построением adjacency и feature view удаляются:

- self-loop edges;
- edges, у которых source и target находятся в разных partitions.

Edges внутри одной partition сохраняются. В частности, test-internal edges образуют отдельный inductive batch. Протокол не называется online forecasting: модель не получает сообщения через границу partition и не обучается на test labels.

## Общее feature space

Transfer использует `structural_7`. Семь столбцов пересчитываются из edgelist и `time_step`:

1. `in_degree`;
2. `out_degree`;
3. `in_unique_neighbors`;
4. `out_unique_neighbors`;
5. `has_incoming`;
6. `has_outgoing`;
7. `step_norm`.

Values пересчитываются отдельно после edge filtering. Нельзя использовать degrees, посчитанные на полном графе, а затем просто скрыть cross-partition edges: это оставило бы информацию из недоступных связей.

Semantic columns `elliptic_txs_features.csv` проверяются validator, но не переносятся напрямую в raw target. У real Elliptic нет гарантированно сопоставимой behavior chain и amount semantics для всех 165 колонок.

## Labels и unknown

В binary task используются только две метки:

- `class=1`: positive, illicit;
- `class=2`: negative, licit.

`unknown` остаётся в графе как unlabeled context. Он не входит в:

- training loss;
- выбор threshold;
- F1;
- Precision@K;
- Recall@K.

Это относится и к synthetic source, и к real target. Нельзя подменять unknown на negative, потому что это меняет prevalence и делает score неправильно интерпретируемым.

## Preprocessing и threshold

Primary preprocessing называется `source_only`:

1. structural features считаются отдельно для каждой partition;
2. scaler обучается на synthetic train partition;
3. тот же scaler применяется к synthetic validation, synthetic test и target partitions;
4. threshold выбирается на labeled synthetic validation;
5. threshold замораживается до scoring на test и real data.

Real labels не участвуют в fit или выборе threshold. Real-trained oracle может присутствовать в report как отдельная sensitivity arm, но не заменяет source-only pipeline.

## Метрики

На test partition для каждого шага `41..49` считаются:

- F1 для positive class;
- Precision@K;
- Recall@K.

Primary ranking использует `K=100`. Если в шаге меньше 100 labeled rows, report публикует `k_effective`. При равенстве score сортировка завершается по возрастанию `txId`, поэтому порядок не зависит от порядка строк в памяти.

Неопределённая метрика записывается как `null`, а не как `NaN`. Макро-агрегат считается основным, micro-агрегат публикуется отдельно. Для acceptance primary GraphSAGE должен иметь coverage всех девяти test steps.

## Model roster

В протоколе есть два diagnostic baseline:

- Random Forest;
- SGC proxy.

SGC proxy использует распространение признаков и logistic regression. Это диагностический baseline, не full GNN.

Optional full backend использует CPU PyG и включает:

- GraphSAGE, primary model;
- GAT;
- GIN.

GraphSAGE использует отдельные branches для исходящих и обратных направлений. GAT и GIN используют исходное направление рёбер. Для всех full models в report публикуются model kind, seed, backend и число threads.

### Backend policy

- `backend=full` требует доступный `torch` и `torch-geometric`. Он никогда не переходит молча на SGC.
- `backend=auto` использует full backend, если он доступен. Без extras он переходит в явно названный `sgc_proxy`.
- `backend=sgc` всегда запускает diagnostic SGC arm.
- Флаг `--sage-only` ограничивает full PyG roster только SAGE. RF и SGC diagnostic остаются, но full roster неполон.

Для real acceptance должны быть выполнены все три full model arms: SAGE, GAT и GIN. Наличие только одного PyG-модельного arm не даёт acceptance eligibility.

## Edge shuffle

Для ablation выполняется deterministic directed edge shuffle внутри partition. Политика проверяет сохранение edge count, membership endpoints и directed in/out degrees, насколько это возможно при заданных ограничениях. Self-loop и запрещённые duplicate pairs не создаются.

Shuffle не обещает сохранить connectivity, количество циклей или конкретный motif count. Для original и shuffled arms используются matched seeds и новые model instances. В report попадают:

- shuffle seed;
- число attempts и swaps;
- changed fraction;
- per-step metrics;
- matched delta;
- paired test и Cohen's dz;
- отдельная unpaired Welch sensitivity.

## Smoke и real acceptance

### Smoke

```bash
python -m kyt_engine.synth validate-real \
  --config configs/generator_behavior.yaml \
  --raw-dir data/raw \
  --out data/synthetic/behavior_run/strict_report_smoke.json \
  --tier smoke \
  --backend auto \
  --seeds 0 \
  --k 100
```

Smoke проверяет загрузку raw, partitions, filtering, labels, baseline metrics и report serialization. Он всегда остаётся diagnostic и не может получить acceptance status.

### Real

```bash
python -m kyt_engine.synth validate-real \
  --config configs/generator_behavior.yaml \
  --raw-dir data/raw \
  --out data/synthetic/behavior_run/strict_report_real.json \
  --tier real \
  --backend full \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --k 100
```

Real acceptance требует одновременно:

- pinned raw fingerprint и 203769 строк в target snapshot;
- совпадение canonical config fingerprint и volume fingerprint;
- frozen seed roster `0..9`, то есть `N >= 10` для real acceptance;
- `backend=full`;
- полный roster SAGE, GAT и GIN;
- primary coverage всех test steps `41..49`;
- `K=100` и frozen model hyperparameters;
- exact optional package versions из acceptance environment.

Ожидаемые package versions для full acceptance:

- `torch==2.2.0`;
- `torch-geometric==2.5.0`;
- `scikit-learn==1.3.0`.

Если окружение не соответствует pinned versions, report получает blocked status. Blocked нельзя читать как провал качества модели.

## Seed semantics

`seed: 72` в `configs/generator_behavior.yaml` является canonical seed для обычного standalone run. Он не заменяет frozen downstream roster.

Для strict experiment seeds `0..9` создаются отдельные synthetic datasets. У каждого seed свои CSV fingerprints и свои derived RNG streams. Edge shuffle seeds также выводятся из experiment seed детерминированно, поэтому report можно повторить.

## Статусы и коды возврата

| Ситуация | Статус в report | Exit code |
|---|---|---|
| diagnostic run завершён | `completed` | `0` |
| real run выполнен, но acceptance не выполнен | `completed` с `target_met=false` или `acceptance_eligible=false` | `2` |
| full backend или acceptance environment недоступны | `blocked` | `3` |
| input, config или report path некорректны | `error` | `4` |

`acceptance.target_met` означает только выполнение численного target `primary_macro_test_f1 > 0.5` в eligible real run. Это не гарантия и не описание качества всех downstream задач.

## Ограничения transfer

- Behavior semantic features не воспроизводят joint distribution Elliptic.
- Общий transfer space использует только structural features, поэтому behavior-specific columns не являются доказательством переноса.
- Удаление cross-partition edges может разорвать continuity behavior motifs. Report описывает disjoint temporal subgraphs, а не forecasting.
- Unknown exclusion не превращает оставшиеся labeled rows в репрезентативную популяцию без проверки prevalence.
- Edge shuffle не сохраняет произвольные motifs и connectivity.
- Один held-out snapshot и фиксированный seed roster не дают универсальной оценки будущих данных.
