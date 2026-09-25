# Разработка

## Рабочее окружение

Проект рассчитан на Python `3.10+`. Базовые зависимости устанавливаются из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Для запуска strict downstream нужен `scikit-learn`. Optional full PyG environment устанавливается отдельно:

```bash
pip install -e ".[downstream]"
pip install -c constraints-gnn.txt -e ".[gnn]"
```

Не устанавливайте optional packages только ради того, чтобы smoke report выглядел полным. Наличие пакета проверяется отдельно, а отсутствие PyG должно приводить к явному diagnostic или blocked status.

## Подготовка входов

Перед `generate` проверьте наличие:

```text
data/volume/volume.npy
```

Файл должен содержать 49 неотрицательных целых весов. Он задаёт только распределение transaction slots по времени. Поведенческие профили, actions и суммы генерируются из config и seed.

Real validation читает три CSV из `data/raw`:

- `elliptic_txs_features.csv`;
- `elliptic_txs_classes.csv`;
- `elliptic_txs_edgelist.csv`.

Raw files являются входом для проверки и никогда не используются как каталог generated output. Report `validate-real` также должен записываться вне `data/raw`.

## Основной рабочий цикл

### 1. Генерация

```bash
python -m kyt_engine.synth generate \
  --config configs/generator_behavior.yaml
```

Проверьте, что config содержит `seed: 72`, `graph.mode: behavior` и `features.mode: semantic`. Не меняйте output path на `data/raw` и не добавляйте в него старые CSV вручную.

### 2. Contract validation

```bash
python -m kyt_engine.synth validate \
  --dir data/synthetic/behavior_run
```

Validation проверяет не только размеры CSV. Она также сверяет hashes, feature semantics, edge attribute row order, event causality, entity references, instance membership, chain records, drift realization и unknown policy.

### 3. Smoke downstream

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

Smoke report нужен для проверки raw schema, partitions, edge filtering и записи JSON. Его metrics нельзя использовать как real acceptance.

### 4. Real protocol

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

Этот запуск требует pinned raw snapshot, canonical config fingerprint, volume fingerprint и exact optional package versions. Если хотя бы одно условие не выполнено, сохраняется blocked report с причиной.

## Локальные команды Makefile

Makefile содержит короткие обёртки над теми же тремя CLI-командами:

```bash
make generate
make validate
make validate-real
```

`make validate-real` использует default smoke tier и стандартный report path. Для frozen real acceptance всё равно запускайте `python -m kyt_engine.synth validate-real` с явными флагами.

## Локальные проверки

После изменений в behavior path используйте gates проекта:

```bash
make test
make lint
```

`make test` запускает test suite. `make lint` проверяет `ruff check` и форматирование. Mypy не является обязательным локальным gate в базовом окружении: для strict mypy нужны third-party stubs и отдельное typecheck environment.

Не исправляйте generated artifacts вручную, чтобы обойти failed validation. Сначала исправьте config, registry, writer или provenance, затем сгенерируйте bundle заново.

## Проверка детерминизма

Для одного и того же behavior config, volume profile, seed и `out_dir` должны совпадать:

- количество transactions;
- `txId` и `time_step`;
- directed edgelist;
- edge attributes;
- semantic feature matrix;
- `manifest.json` и CSV hashes.

Если сравниваются два разных каталога output, сначала учитывайте, что `out_dir` входит в effective config и может отличаться в manifest. Не используйте wall-clock time, `hash()` или порядок файловой системы как скрытые источники порядка.

## Работа с новым profile

При добавлении profile нужно проверить весь контракт:

1. Добавить profile в allowlisted registry.
2. Описать entity type, class policy, action IDs, counterparty profiles, chain weights и amount policy.
3. Добавить profile в `_KNOWN_PROFILE_IDS` и config validation.
4. Проверить canonical profile ordering.
5. Добавить event и node coverage tests.
6. Проверить late drift eligibility, если profile является novel.
7. Проверить instance membership и edge endpoint references.
8. Обновить manifest validator и документацию.

Нельзя закрыть нехватку transactions добавлением случайного background node, generic edge или profile без action. Scheduler обязан либо выполнить quota, либо завершиться с ошибкой.

## Работа с новым action

Action должен иметь явные:

- actor profiles;
- counterparty profiles;
- target class policy;
- edge kind;
- amount policy;
- channel.

После добавления проверьте non-bridge chain locality, bridge linkage, source-before-target causality, class mapping и referential coverage в manifest. Если action требует нового внешнего payload, сначала определите его typed schema и запретите неизвестные поля.

## Работа с partitions

При изменении downstream partition logic нельзя переносить adjacency, построенную до фильтрации. Сначала определите partition по `time_step`, затем удалите cross-partition и self-loop edges, затем пересчитайте structural features.

`unknown` должен оставаться unlabeled. Если появляется новый metric или threshold, явно укажите, использует ли он unknown. Default для binary F1 и P@K/R@K: не использует.

Не меняйте fixed ranges `1..30`, `31..40`, `41..49` без изменения protocol version и acceptance report. Random split не является заменой temporal inductive split.

## Optional full PyG

Full backend состоит из CPU GraphSAGE, GAT и GIN. Установка extras сама по себе не означает, что acceptance environment pinned. Проверяйте:

- доступность импортов и probe;
- `is_full_gnn` в report;
- package versions;
- `backend=full` без fallback;
- полный model roster;
- primary SAGE coverage по всем test steps.

`backend=auto` без extras допустим для smoke, но не для real acceptance. SGC proxy всегда остаётся diagnostic baseline.

## Частые ошибки

| Симптом | Что проверить |
|---|---|
| Generator не стартует | наличие `data/volume/volume.npy`, его форму и seed в config |
| Quota mismatch | сумма `tx_quota` должна равняться `n_txs` |
| Drift error | phases должны покрывать `1..49` без gaps и overlaps |
| CSV hash mismatch | файл изменён после записи или bundle собран из разных run |
| Edge attribute error | число строк и порядок endpoints должны совпадать с edgelist |
| Behavior validation error | удалённый или изменённый event/entity/instance reference |
| Real report blocked | raw fingerprint, config fingerprint, volume fingerprint, PyG packages и seed roster |
| Report path error | `--out` нельзя размещать внутри `data/raw` |
| F1 ниже target | сначала прочитать `status`, `acceptance_eligible` и `target_met`; blocked не является performance measurement |

## Ограничения разработки

- Behavior policies задаются для контролируемой симуляции и не являются измеренной активностью реальных пользователей.
- Semantic feature space нельзя напрямую сравнивать с полными 165 Elliptic features как с одной и той же Joint distribution.
- Event ledger не даёт полной экономической conservation или motif-level attestation.
- Chain metadata не заменяет native transaction parser.
- Acceptance зависит от raw snapshot и frozen protocol. Изменение raw data, config или dependency versions требует нового protocol fingerprint и нового real run.
