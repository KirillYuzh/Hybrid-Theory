# 02 Subagent — Анализ Spillety и что нужно от генератора синтетических транзакций

Отчёт subagent-2 (анализ Spillety). Полный текст в сессии; здесь тезисы.

## 1. Что такое Spillety

- KYT/AML: санкционные якоря → индуктивный графовый энкодер (GraphSAGE/EvolveGCN) → retrieval (HNSW/PQ) → контрастивные эмбеддинги → entity resolution (CIOH+временная близость+degree similarity) → GBDT-калибровка → evidence worm audit + SAR.
- Метрики: PR-AUC 0.656, ECE 0.011, precision@100=1.0 (тест 41–49, Elliptic). Retrieval recall@10 0.995 (HNSW vs brute). Потолок tabular+PCA ~0.65–0.70 на поздних шагах.
- Данные: только Elliptic (203.7k tx, 167 фич, 234k рёбер) + санкции OFAC-SDN/EU-FSF 507 адресов (`data/sanctions/addresses.csv`).
- Валидация: строгая temporal / walk-forward (train ≤30, valid 31–40, test 41–49).
- Инфраструктура (Ansible k3s MinIO Iceberg Spark) реальна, но ДЕМО-СЕРВИСА И FASTAPI НЕТ.

## 2. Слабые места / открытые проблемы Spillety

- Один датасет (Elliptic) — мало данных для контрастивного обучения, аугментации редких схем, проверки обобщения на новых санкциях.
- Ground truth только по «вход перевода» рёбрам; нет размеченных схем (fan-out, peel-chain, mixer) как якорей.
- Калибровка τ и операционные метрики (ttd, alert_to_sar, cost/alert) требуют «реальной» частоты алертов — нет данных для валидации.
- PQ (кодирование) и HNSW требуют стресс-теста на распределениях, которых нет в Elliptic.
- Causal-фильтр (E-value, pass_rate test 0.750) нуждается в контролируемых данных для проверки.
- Уже есть: `temporal/policy.py` (drift/novel/noise/normal regimes) — нечего прогонять, кроме Elliptic.

## 3. Идея «генератор синтетических транзакций» — критичная оценка

### Польза
- **Ground-truth бенчмарк retrieval**: знаем, какие адреса «вход в перевод» и какие схемы размечены — можно точно мерить recall@10/PR-AUC HNSW/PQ против «правильного» ответа.
- **Аугментация редких классов**: посадка известных схем (mixer, peel-chain, fan-out, wash-trade) с контролем сколько и каких — чего в Elliptic нет.
- **Якоря для контрастивного обучения**: генерация санкционных адресов со «знанием» об их участии в схемах — положительные пары точны.
- **Стресс-тест temporal shift**: внесение дрейфа (закрытие даpкnet-площадки = аналог шага 43) под контролем — проверка drift-гейтов и walk-forward.
- **Калибровка τ / операционные метрики**: известная базовая частота illicit → честный расчёт optimal τ, FP-нагрузки, TTD.

### Требования к генератору (контракты Spillety)
1. **Формат Elliptic++** (главный контракт `spillety/data/loader.py:6-45`): features `txId, time_step, feat_2..feat_166`; classes `txId, class ∈ {1,2,unknown}`; edgelist `txId1, txId2` (рёбра «вход перевода»).
2. **Account graph** (`embeddings/account_graph.py:19-57`): tx = `Sequence[Sequence[Hashable]]` входных адресов + флаги `is_coinjoin`/`is_exchange_hot`.
3. **Санкционные якоря**: `data/sanctions/addresses.csv` (address, currency, source, list, date).
4. **Временная целостность**: time_step 1..49, когерентность для walk-forward (split ≤30/31–40/41–49), возможность внесения режимных сдвигов.
5. **Честные лейблы** ground truth + манифест (sha256), как уже делает Spillety для санкций.

## 4. Что переиспользуемо из Hybrid-Theory

- `data/synthetic.py`, `training/agent_based.py`, `simulations/`, `filtering/` — заготовки генерации, но формат Ethereum, НЕ совпадает с контрактом Spillety (нужна адаптация к Elliptic++ схеме).
- `features/*` — 191 фича (166 stat + 26 behavior) — могут стать основой для генерации реалистичных значений фич.

## 5. Ключевой конфликт форматов

- Hybrid-Theory: `data/elliptic.py` ожидает nodes/edges/classes с колонкой `label`;
- Spillety: `load_elliptic` ожидает feat_2..feat_166 + class ∈ {1,2,unknown} + edgelist.
- Генератор должен выдавать СХЕМУ Spillety, а не схему Hybrid-Theory.