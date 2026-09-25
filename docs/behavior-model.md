# Модель поведения

## Основные объекты

Симулятор поведения разделяет данные на четыре уровня.

| Объект | Что означает |
|---|---|
| `Entity` | типизированный участник с profile, chain и состоянием последней транзакции |
| `Action` | разрешённое действие с actor policy, counterparty policy, channel и amount policy |
| `Event` | одна причинная запись, которая создаёт одну целевую транзакцию |
| `Instance` | явная компонента связанных events с общей provenance-записью |

Транзакционный граф строится из событий, а не из заранее размеченного шаблона. Поэтому одно событие может иметь несколько участников, но в CSV оно всё равно представлено одной строкой с одной меткой уровня действия.

## Профили

В реестре используются следующие профили.

| `profile_id` | Тип сущности | Политика класса | Роль |
|---|---|---|---|
| `ordinary_wallet` | `individual` | licit | обычный перевод или bootstrap |
| `exchange_hub` | `exchange` | licit | входящие клиентские переводы и исходящие переводы разрешённым получателям |
| `miner_payout` | `miner` | licit | регулярные payout events |
| `wallet_provider` | `wallet_provider` | licit | ограниченные по дисперсии micropayments |
| `individual` | `individual` | licit | редкие переводы среднего размера |
| `mixer` | `illicit_actor` | illicit | ограниченный fan-in/fan-out scaffold |
| `wash_round_trip` | `illicit_actor` | illicit, novel | поздний repeated-entity scaffold |

Профиль задаёт только разрешённую политику. Реальные chain weights, activity weights и amount policies являются детерминированными настройками симуляции, а не измеренными частотами Elliptic.

## Действия и контрагенты

Каждое событие выбирает действие из списка разрешённых для профиля. Контрагент также выбирается только из профилей, перечисленных в политике этого действия. Такой список не позволяет случайному узлу или неизвестному типу сущности появиться в графе без provenance.

В реестре есть следующие действия:

- `transfer`;
- `hub_in`;
- `hub_out`;
- `payout`;
- `micropayment`;
- `mix_in`;
- `mix_out`;
- `round_trip`;
- `bridge_link`.

Обычные действия используют канал `on_chain`. `bridge_link` использует отдельный канал `bridge` и допускается только для явно cross-chain события. Для обычного действия все участники должны относиться к одной сети. Если случайно выбранный контрагент находится в другой сети, действие заменяется на разрешённое bridge action, а не создаётся неявное cross-chain ребро.

## Жизненный цикл события

Обработка одного слота выполняется в следующем порядке:

1. Выбирается допустимый профиль с учётом фазы времени.
2. Выбираются действие, actor и counterparty.
3. Определяется причинная исходная транзакция, если у выбранной сущности уже есть предыдущая транзакция.
4. Резервируется новый локальный `target_tx_id`.
5. В журнал добавляются ссылки на источник, целевую сущность, действие, chain и amount.
6. Обновляется состояние сущностей.
7. Событие атомарно связывается с одним целевым node и, при наличии источника, с одним направленным ребром.

Событие без исходного ребра допустимо. Это явная запись bootstrap с пустым `source_tx_ids`, а не скрытый filler. Если источник есть, должны выполняться условия:

- исходная транзакция уже существует;
- `source.step <= target.step`;
- event ID источника меньше event ID цели;
- source entity совпадает с target entity предыдущего события;
- в `edgelist` есть соответствующее направленное ребро.

## Instances и их границы

Instances строятся по связности событий через source edges. Они не выводятся из диапазона contiguous IDs. Поэтому transactions разных instances могут идти вперемешку в CSV.

Для каждого instance manifest хранит:

- `instance_id`;
- `event_ids`;
- `node_ids`;
- `edge_ids`;
- `entity_ids`;
- `profile_ids`;
- `temporal_window`;
- `chain_path`.

Проверка instance требует, чтобы все его edges имели endpoints внутри его `node_ids`. Это не гарантирует, что поведение полностью соответствует экономике mixer или wash trading. Ledger подтверждает причинную и структурную provenance, но не заменяет отдельную motif-level калибровку.

## Метки и unknown

CSV хранит class строки:

- `1` означает illicit action target;
- `2` означает licit action target;
- `unknown` оставляется допустимым значением Elliptic-контракта.

В текущем наборе профили имеют policy `licit` или `illicit`, поэтому `unknown` не используется как скрытая licit-negative метка. Для multi-party event class в CSV задаётся policy целевого action. Classes всех участников и их entity references остаются в `manifest.json`.

В downstream `unknown` означает unlabeled context. Такие строки не участвуют в loss, выборе threshold, F1, Precision@K и Recall@K. Это решение не превращает unknown в negative.

## Chain metadata

В manifest поддерживаются три synthetic chain profiles:

| Ключ | Chain family | Native asset | Decimals |
|---|---|---|---|
| `bitcoin` | `utxo` | BTC | 8 |
| `ethereum` | `account` | ETH | 18 |
| `tron` | `account` | TRX | 6 |

`chain_id`, synthetic native transaction IDs и synthetic native entity IDs являются manifest metadata. Они не добавляются в четыре CSV и не выглядят как реальные адреса.

CSV `amount` является нормализованной синтетической величиной. `native_amount_minor` не заполняется. Bridge record описывает typed link между логическими chain records, но не утверждает, что существует полный native bridge ledger, cross-asset balance или атомарная транзакция на двух сетях.

## Временные фазы

Behavior-only расписание покрывает все шаги `1..49` без пропусков и пересечений:

| Фаза | Шаги | Поведение |
|---|---|---|
| `stable` | `1..39` | обычная политика illicit активна |
| `shutdown` | `40..44` | старые illicit профили hard-suppressed |
| `novel` | `45..49` | разрешён только профиль из `novel_profile_ids` |

В `manifest.behavior.drift_schedule` сохраняются и конфигурация фаз, и counts, которые получились для конкретного seed. Эти counts не являются оценками реальной частоты фаз.

## Seed namespaces

Canonical seed для обычного behavior run равен `72`. Поведенческий генератор не использует один общий RNG state. Для каждого namespace создаётся отдельный PCG64 generator из seed и namespace:

```text
sha256(f"{seed}:{namespace}")[0:8] -> little-endian uint64
```

Используются только следующие namespaces:

- `behavior:population`: создание сущностей и назначение chain;
- `behavior:schedule`: выбор step slots и profile assignment;
- `behavior:transition`: выбор actor, action и counterparty;
- `behavior:amount`: выбор синтетической суммы;
- `behavior:drift`: eligibility и phase gate.

Порядок registry, profiles и records канонизируется до обращения к RNG. `hash()`, порядок файловой системы и wall-clock time не являются источниками случайности.

## Что модель не утверждает

- Семантические признаки не воспроизводят joint distribution реального Elliptic.
- Activity weights и amount policies не являются измеренной калибровкой.
- Bootstrap events не доказывают происхождение средств.
- Mixer scaffold не гарантирует conservation всех входов и выходов.
- Round-trip scaffold не является доказательством полного motif closure.
- Cross-chain record не является полным bridge или asset ledger.
- Наличие class `1` в synthetic ground truth не означает, что такой же риск подтверждён на реальных данных.
