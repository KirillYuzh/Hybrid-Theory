# Hybrid Theory (KYT Engine): Методология

Техническая документация

---

## 1. Конвейер инженерии признаков

Конвейер инженерии признаков преобразует сырые записи о Bitcoin-транзакциях в 191 числовой признак, пригодный для классификации методами машинного обучения. Конвейер реализован в `src/kyt_engine/features/` и состоит из двух этапов: базовые статистические признаки (165) и поведенческие признаки (26). В production-версии дополнительно вычисляются 4 графовых признака и 64-мерные Node2Vec эмбеддинги, что даёт 196 + 64-d = 260-мерный вектор (см. §6).

### 1.1 Предобработка данных

Сырые данные о транзакциях загружаются из CSV-файлов (`nodes.csv`, `edges.csv`, `classes.csv`) и объединяются по `txId`. Перед извлечением признаков применяются следующие этапы предобработки:

1. **Приведение типов:** `timestamp`, `value`, `gas_price`, `gas_used`, `block_number` конвертируются в числовые типы с `errors="coerce"`
2. **Заполнение пропущенных значений:** NaN-значения заполняются 0.0 для числовых столбцов
3. **Временная декомпозиция:** Временные метки декомпозируются на компоненты `hour` (0–23), `day_of_week` (0–6) и `date` (ГГГГ-ММ-ДД)
4. **Вычисление интервалов:** Интер-транзакционные интервалы вычисляются по адресу как `timestamp.diff().clip(lower=0.0)`
5. **Сортировка:** Данные сортируются по `(address, timestamp)` для обеспечения корректного временного порядка

### 1.2 Категории базовых признаков

#### 1.2.1 Статистика стоимости (26 признаков)

Эти признаки характеризуют распределение стоимости транзакций для каждого адреса:

| Признак | Формула | Описание |
|---------|---------|-------------|
| `value_mean` | $\bar{v} = \frac{1}{n}\sum_{i=1}^{n} v_i$ | Средняя стоимость транзакции |
| `value_std` | $\sigma_v$ | Стандартное отклонение |
| `value_min` | $\min(v)$ | Минимальное значение |
| `value_max` | $\max(v)$ | Максимальное значение |
| `value_median` | $\text{median}(v)$ | Медианное значение |
| `value_q25` | $Q_{25}(v)$ | 25-й процентиль |
| `value_q75` | $Q_{75}(v)$ | 75-й процентиль |
| `value_skew` | $\gamma_1(v)$ | Асимметрия (Фишера) |
| `value_kurtosis` | $\kappa(v)$ | Эксцесс |
| `value_range` | $\max(v) - \min(v)$ | Размах |
| `value_iqr` | $Q_{75} - Q_{25}$ | Межквартильный размах |
| `value_sum` | $\sum v_i$ | Суммарная стоимость |
| `value_cv` | $\sigma_v / \bar{v}$ | Коэффициент вариации |
| `value_log_mean` | $\overline{\ln(1+v)}$ | Среднее логарифмированных значений |
| `value_log_std` | $\sigma_{\ln(1+v)}$ | Стд логарифмированных значений |
| `value_entropy` | $H(v) = -\sum p_j \log_2 p_j$ | Дискретизированная энтропия (20 бинов) |
| `value_positive_ratio` | $\frac{1}{n}\sum \mathbb{1}[v_i > 0]$ | Доля положительных значений |
| `value_zero_ratio` | $\frac{1}{n}\sum \mathbb{1}[v_i = 0]$ | Доля нулевых значений |
| `value_max_min_ratio` | $\max(v) / \min(v)$ | Отношение максимума к минимуму |
| `value_skew_mean` | $\gamma_1 \cdot \bar{v}$ | Взвешенное асимметрией среднее |
| `value_upper_outlier_ratio` | Доля выше $Q_{75} + 1.5 \cdot \text{IQR}$ | Доля верхних выбросов |
| `value_lower_outlier_ratio` | Доля ниже $Q_{25} - 1.5 \cdot \text{IQR}$ | Доля нижних выбросов |
| `value_concentration` | $\max(v) / \sum v$ | Максимальная концентрация |
| `value_dominance` | $\sum_{i=1}^{5} v_{(i)} / \sum v$ | Доля топ-5 |
| `value_small_ratio` | Доля ниже $0.1 \cdot \text{median}(v)$ | Доля мелких транзакций |
| `value_large_ratio` | Доля выше $10 \cdot \text{median}(v)$ | Доля крупных транзакций |

#### 1.2.2 Статистика газа (26 признаков)

Идентичный набор статистических признаков, вычисляемых для `gas_price` вместо `value`. Стоимость газа отражает приоритет комиссии транзакции и является сильным поведенческим сигналом — нелегитимные транзакции часто используют нетипичные стратегии газа.

#### 1.2.3 Статистика интервалов транзакций (16 признаков)

Признаки, производные от интер-транзакционных временных разрывов (в секундах):

| Признак | Описание |
|---------|-------------|
| `interval_mean` | Средний интервал |
| `interval_std` | Стандартное отклонение интервалов |
| `interval_min` | Минимальный интервал |
| `interval_max` | Максимальный интервал |
| `interval_median` | Медианный интервал |
| `interval_q25` / `interval_q75` | Границы межквартильного размаха |
| `interval_skew` / `interval_kurtosis` | Форма распределения |
| `interval_range` | Размах интервалов |
| `interval_cv` | Коэффициент вариации |
| `interval_log_mean` / `interval_log_std` | Логарифмированные статистики |
| `interval_entropy` | Дискретизированная энтропия |
| `interval_burstiness` | Доля интервалов ниже $0.1 \cdot \text{median}$ |
| `interval_regularity` | $1 - \min(\sigma/\mu, 1)$ |

#### 1.2.4 Признаки времени суток (10 признаков)

| Признак | Описание |
|---------|-------------|
| `hour_mean` / `hour_std` | Статистики распределения часа |
| `hour_entropy` | Энтропия Шеннона по 24 часовым бинам |
| `night_ratio` | Доля транзакций в часы 0–5, 22–23 |
| `morning_ratio` | Доля в часы 6–11 |
| `afternoon_ratio` | Доля в часы 12–17 |
| `evening_ratio` | Доля в часы 18–21 |
| `peak_hour` | Час с максимальной активностью |
| `hour_concentration` | Максимальная доля часа |
| `hour_bimodality` | $1 - \sigma_h / \mu_h$ для ненулевых часов |

#### 1.2.5 Признаки дня недели (8 признаков)

| Признак | Описание |
|---------|-------------|
| `dow_entropy` | Энтропия Шеннона по 7 дневным бинам |
| `weekend_ratio` | Доля на субботу/воскресенье |
| `midweek_ratio` | Доля на пн/вт/ср |
| `endweek_ratio` | Доля на чт/пт/сб |
| `dow_concentration` | Максимальная доля дня |
| `dow_regularity` | Доля активных дней / 7 |
| `day_span` | Охват календарных дней |

#### 1.2.6 Сетевые метрики (18 признаков)

Признаки теории графов, вычисляемые из локального окружения каждого адреса:

| Признак | Описание |
|---------|-------------|
| `in_degree` | Количество уникальных отправителей |
| `out_degree` | Количество уникальных получателей |
| `total_degree` | Общее количество уникальных партнёров |
| `in_out_ratio` | $d_{\text{in}} / d_{\text{out}}$ |
| `unique_in` / `unique_out` / `unique_total` | Уникальные контрагенты |
| `degree_concentration` | Доля ведущего партнёра |
| `in_value_ratio` / `out_value_ratio` | Направление потока стоимости |
| `mutual_ratio` | Двусторонние партнёры / общее число |
| `reciprocity` | Доля взаимных пар с фактическими двусторонними рёбрами |
| `hub_score` | $d_{\text{out}} / d_{\text{total}}$ |
| `authority_score` | $d_{\text{in}} / d_{\text{total}}$ |
| `pagerank_approx` | Аппроксимация как $\text{in\_value} / \text{total\_value}$ |
| `star_ratio` | Концентрация ведущего партнёра |

#### 1.2.7 Анализ контрагентов (12 признаков)

| Признак | Описание |
|---------|-------------|
| `unique_counterparties` | Количество различных контрагентов |
| `counterparty_concentration` | Доля ведущего контрагента |
| `top_counterparty_ratio` | Аналогично выше |
| `top5_counterparty_ratio` | Доля топ-5 контрагентов |
| `new_counterparty_ratio` | Доля впервые увиденных контрагентов |
| `return_counterparty_ratio` | Доля возвращающихся контрагентов |
| `counterparty_reciprocity` | Доля двусторонних контрагентов |
| `counterparty_entropy` | Энтропия Шеннона распределения контрагентов |
| `counterparty_churn` | Частота смены партнёров |
| `stable_counterparty_ratio` | Партнёры с количеством > 1 |
| `bridge_counterparty_ratio` | Партнёры, присутствующие и во входящих, и в исходящих |
| `counterparty_hhi` | Индекс Херфиндаля-Хиршмана: $\sum (f_j / F)^2$ |

#### 1.2.8 Признаки тренда стоимости (11 признаков)

| Признак | Описание |
|---------|-------------|
| `value_trend_slope` | Наклон линейной регрессии |
| `value_trend_r2` | $R^2$ линейной аппроксимации |
| `value_momentum_{3,5,10}` | $\bar{v}_{\text{last } k} - \bar{v}_{\text{rest}}$ |
| `value_acceleration` | Средняя разность второго порядка |
| `value_jerk` | Средняя разность третьего порядка |
| `value_volatility_{3,5,10}` | Скользящее.std по окну $k$ |
| `trend_consistency` | $\sigma_{\text{rolling-3}} / \mu_{\text{rolling-3}}$ |
| `value_max_streak` | Наибольшая монотонная последовательность / $n$ |

#### 1.2.9 Признаки тренда газа (7 признаков)

Аналогичные трендовые признаки для `gas_price`, включая `gas_efficiency_trend` (наклон `gas_used / gas_price` во времени) и `gas_spike_freq` (доля шагов, когда газ превышает 3x предыдущего).

#### 1.2.10 Признаки автокорреляции (15 признаков)

| Признак | Описание |
|---------|-------------|
| `value_acf_{1,2,3,5,10}` | ACF стоимости на лагах 1, 2, 3, 5, 10 |
| `gas_acf_{1,2,3}` | ACF gas_price на лагах 1, 2, 3 |
| `interval_acf_{1,2,3}` | ACF интервалов на лагах 1, 2, 3 |
| `acf_decay_rate` | $1 - |\text{ACF}(k^*)| / |\text{ACF}(1)|$ где $k^*$ — первый лаг с $|\text{ACF}| < 0.5 \cdot |\text{ACF}(1)|$ |
| `acf_sign_changes` | Доля смен знака в первых 5 ACF-значениях |
| `periodicity_score` | $\max(0, \text{ACF}(2) - \text{ACF}(1)^2)$ |
| `stationarity_score` | $R / \sigma$ где $R$ — размах кумулятивных отклонений |
| `hurst_exponent` | Экспонента Хёрста через анализ R/S: $H = \text{slope}(\log k, \log \tau(k))$ |

#### 1.2.11 Лаговые признаки (13 признаков)

| Признак | Описание |
|---------|-------------|
| `value_lag_{1,2,3}` | Автокорреляция стоимости на лагах 1, 2, 3 |
| `gas_lag_{1,2,3}` | Автокорреляция газа на лагах 1, 2, 3 |
| `value_gas_corr` | Корреляция Пирсона между стоимостью и газом |
| `value_block_corr` | Корреляция между стоимостью и номером блока |
| `gas_block_corr` | Корреляция между газом и номером блока |
| `value_ts_corr` | Корреляция между стоимостью и временной меткой |
| `gas_ts_corr` | Корреляция между газом и временной меткой |
| `interval_value_corr` | Корреляция между интервалом и следующей стоимостью |

#### 1.2.12 Блоковые признаки (4 признака)

| Признак | Описание |
|---------|-------------|
| `block_span` | $\max(\text{blocks}) - \min(\text{blocks})$ |
| `blocks_per_tx` | Уникальные блоки / общее число транзакций |
| `unique_blocks_ratio` | Аналогично выше |
| `block_reuse_ratio` | $1 - |\text{unique block counts}| / |\text{unique blocks}|$ |

### 1.3 Поведенческие признаки (26 признаков)

#### 1.3.1 Скорость реакции (3 признака)

Измеряют отзывчивость адреса на входящие транзакции:

- **`reaction_speed_mean`**: Среднее время между последовательными транзакциями, где направление меняется (вход → исход или исход → вход)
- **`reaction_speed_median`**: Медианное время реакции
- **`fast_reaction_ratio`**: Доля времён реакции ниже $0.5 \cdot \text{median}$

#### 1.3.2 Циркадные ритмы (5 признаков)

| Признак | Формула |
|---------|---------|
| `circadian_regularity` | $1 - \sigma_h / 12$ |
| `peak_activity_hour` | $\arg\max_h \text{count}(h)$ |
| `nocturnal_ratio` | $\sum_{h \in \text{night}} c(h) / C$ |
| `work_hours_ratio` | $\sum_{h=9}^{16} c(h) / C$ |
| `circadian_amplitude` | $(c_{\max} - c_{\min}) / C$ |

#### 1.3.3 Стратегия газа (6 признаков)

| Признак | Описание |
|---------|-------------|
| `gas_aggressiveness` | Средний процентиль ранга стоимости газа |
| `gas_bid_ratio` | Доля выше глобальной медианы газа |
| `gas_patience` | Доля ниже глобальной медианы газа |
| `gas_optimization_score` | $1 - \min(\sigma / \mu, 1)$ |
| `gas_predictability` | $\max(0, \text{ACF}_1)$ последовательности газа |
| `mean_gas_percentile` | Средний ранг / $n$ |

#### 1.3.4 Поведение интервалов транзакций (3 признака)

| Признак | Описание |
|---------|-------------|
| `burst_ratio` | Доля ниже 25-го процентиля |
| `long_pause_ratio` | Доля выше $3 \cdot \text{median}$ |
| `rapid_fire_ratio` | Доля ниже $0.1 \cdot \text{median}$ |

#### 1.3.5 Энтропия Шеннона (4 признака)

| Признак | Описание |
|---------|-------------|
| `value_distribution_entropy` | Дискретизированная энтропия стоимостей (20 бинов) |
| `behavioral_counterparty_entropy` | $H(\text{контрагенты}) = -\sum p_j \log_2 p_j$ |
| `temporal_entropy` | $H(\text{часы})$ по 24-часовому распределению |
| `behavioral_gas_entropy` | Дискретизированная энтропия стоимости газа |

#### 1.3.6 Разнообразие контрагентов (5 признаков)

| Признак | Описание |
|---------|-------------|
| `counterparty_diversity` | Уникальные контрагенты / общее число транзакций |
| `counterparty_growth_rate` | $\bar{n}_{\text{new}}^{\text{2nd half}} - \bar{n}_{\text{new}}^{\text{1st half}}$ |
| `counterparty_stability` | Партнёры с количеством > 1 / уникальные партнёры |
| `new_vs_returned_ratio` | Новое / возвращённое |
| `counterparty_herfindahl` | $1 - \sum (f_j / F)^2$ |

---

## 2. Модель LightGBM

### 2.1 Гиперпараметры

Классификатор LightGBM настроен со следующими параметрами:

```python
LGBMClassifier(
    n_estimators=500,        # Количество раундов бустинга
    learning_rate=0.05,      # Размер шага сжатия
    max_depth=-1,            # Без ограничения глубины дерева
    num_leaves=63,           # Максимум листьев на дерево (2^6 - 1)
    min_child_samples=20,    # Минимум сэмплов в листовом узле
    subsample=0.8,           # Доля подвыборки строк
    colsample_bytree=0.8,    # Доля подвыборки столбцов на дерево
    reg_alpha=0.1,           # L1-регуляризация
    reg_lambda=1.0,          # L2-регуляризация
    class_weight="balanced", # Автокоррекция дисбаланса
    random_state=42,         # Воспроизводимость
    verbose=-1,              # Подавление логов
    n_jobs=1,                # Однопоточность
)
```

### 2.2 Процедура обучения

1. **Подготовка признаков:** Замена бесконечностей на NaN, заполнение NaN значением 0.0
2. **Обучение модели:** Стандартный градиентный бустинг с целью log-loss
3. **Калибровка:** Изотоническая регрессия по вероятностям на выделенном калибровочном наборе
4. **Оптимизация порога:** Поиск по сетке по $\tau \in [0.1, 0.9)$ с шагом 0.01, максимизация:
$$F_1(\tau) = \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$

### 2.3 Калибровка вероятностей

Сырые вероятности LightGBM калибруются с помощью `sklearn.isotonic.IsotonicRegression` с `out_of_bounds="clip"`. Калибровщик обучается либо на отдельном калибровочном наборе (`X_cal`, `y_cal`), либо, если он не предоставлен, на самих обучающих данных. Это обеспечивает хорошую калиброванность предсказанных вероятностей и их пригодность для принятия решений на основе порогов.

---

## 3. Детектор Autoencoder

### 3.1 Архитектура

Autoencoder (автоэнкодер) — это симметричная полносвязная сеть:

```
Вход (d=191) → Linear(d, enc_dim) → ReLU → Linear(enc_dim, 32) → ReLU
                                                          ↓
                                                Узкое горлышко (32 изм.)
                                                          ↓
32 → Linear(32, enc_dim) → ReLU → Linear(enc_dim, d) → Выход (d=191)
```

Где `enc_dim = max(64, d // 3)`.

### 3.2 Обучение

- **Обучающие данные:** Только легитимные транзакции (класс 0)
- **Функция потерь:** Среднеквадратичная ошибка (MSE) между входом и реконструкцией
- **Оптимизатор:** Adam с $lr = 10^{-3}$
- **Эпохи:** 100
- **Размер батча:** 256
- **Нормализация:** Z-score масштабирование, вычисленное на легитимных обучающих данных

### 3.3 Скоринг аномалий

При выводе скор аномалии для сэмпла $x$ вычисляется как:
$$s(x) = \frac{1}{d} \|x - \hat{x}\|^2$$

Нормализованный скор:
$$\hat{s}(x) = \min\left(\frac{s(x)}{s_{\text{threshold}}}, 1.0\right)$$

где $s_{\text{threshold}}$ — $(1 - \text{contamination})$-й квартиль ошибок реконструкции на обучающих данных (contamination = 0.05).

---

## 4. Вариационный Autoencoder (VAE)

### 4.1 Архитектура

Кодировщик VAE генерирует векторы среднего $\mu$ и логарифмической дисперсии $\log\sigma^2$, из которых латентные сэмплы извлекаются через репараметризацию:

$$z = \mu + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$$

```
Вход (d) → Enc(d, enc_dim) → ReLU → Enc(enc_dim, enc_dim//2) → ReLU
                                                                ↓
                                                      μ: Linear(enc//2, 16)
                                                      logσ²: Linear(enc//2, 16)
                                                                ↓
                                                z ~ N(μ, σ²I) [репараметризация]
                                                                ↓
Dec(16, enc//2) → ReLU → Dec(enc//2, enc_dim) → ReLU → Dec(enc_dim, d)
```

### 4.2 Цель обучения

$$\mathcal{L}(x) = \|x - \hat{x}\|^2 + D_{\text{KL}}(q(z|x) \| \mathcal{N}(0, I))$$

$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{16} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### 4.3 Скор аномалии

$$s(x) = e_{\text{recon}}(x) + e_{\text{KL}}(x)$$

где $e_{\text{recon}} = \frac{1}{d}\|x - \hat{x}\|^2$ и $e_{\text{KL}} = -\frac{1}{2}\sum_j(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$.

---

## 5. Стекинг-ансамбль

### 5.1 Мета-обучатель

$$P(y=1|x) = \sigma\left(\beta_0 + \beta_1 \cdot p_{\text{LGBM}}(x) + \beta_2 \cdot p_{\text{AE}}(x)\right)$$

Мета-обучатель — `LogisticRegression(max_iter=1000, class_weight="balanced")`, обученный на стекинговых вероятностных выходах LightGBM и Autoencoder на обучающем наборе.

### 5.2 Опциональный поведенческий сигнал

Ансамбль поддерживает опциональный третий вход: поведенческий вероятностный скор (например, от детектора на основе правил). При его наличии:

$$P(y=1|x) = \sigma\left(\beta_0 + \beta_1 p_{\text{LGBM}} + \beta_2 p_{\text{AE}} + \beta_3 p_{\text{behav}}\right)$$

### 5.3 Выбор порога

Оптимальный порог классификации выбирается путём максимизации $F_1$-меры по предсказаниям на обучающем наборе с помощью поиска по сетке по $\tau \in [0.1, 0.9)$ с шагом 0.01.

---

## 6. Распределённая инженерия признаков (Spark)

В production-версии вычисление признаков выполняется распределённо через Spark Structured Streaming на Iceberg-таблицах. Каждый экстрактор реализует Spark-совместимый интерфейс `transform(df) -> df`.

### 6.1 StatFeatureExtractor (Spark Streaming)

**Контракт:** принимает `DataFrame` из Kafka-топика `raw_txs`, возвращает DataFrame с 166 статистическими признаками (см. §1.2.1–1.2.12). Аггрегирует по `(from_address, to_address)` с окном 30 дней.

**Метод:** `df.withColumns({f"stat_feat_{i}": ... for i in range(1, 167)})`. Обрабатывает edge cases: `null → 0`, `inf → 0`, пустые группы → вектор нулей.

### 6.2 BehaviorFeatureExtractor (Spark Batch)

**Контракт:** принимает DataFrame из Iceberg-таблицы `features` (последние 90 дней), возвращает DataFrame с 26 поведенческими признаками на адрес.

**Окна:** 30d, 60d, 90d. Поведенческие фичи требуют длинной истории и поэтому вычисляются в batch-режиме ежечасно.

**Метод:** `compute_behavioral(df) -> df` с `behavior_feat_1...26` (см. §1.3).

### 6.3 GraphFeatureExtractor (Spark Batch)

**Контракт:** принимает рёбра из `raw_txs` за последние 30 дней, возвращает 4 графовых признака на адрес.

**Алгоритмы:** PageRank (15 итераций, damping=0.85), in/out degree через `groupBy().count()`, triangle count через GraphFrame `triangleCount()`, clustering coefficient через `localClusteringCoefficient()`.

### 6.4 EmbeddingGenerator (Spark Batch)

**Контракт:** принимает GraphFrame рёбер, возвращает 64-мерные Node2Vec эмбеддинги.

**Параметры:**
- `walk_length = 40`
- `num_walks = 10` на узел
- `dimensions = 64`
- `p = 1.0, q = 1.0` (баланс BFS/DFS)
- `window = 10`
- `epochs = 5`

**Метод:** `train_node2vec(edges_df) -> df` с колонками `embedding_1...64`. Узлы вне графа получают нулевой вектор.

### 6.5 SparkApplication (Kubernetes)

```yaml
executor:
  instances: 10
  cores: 4
  memory: "8g"
driver:
  cores: 2
  memory: "4g"
sparkConf:
  spark.sql.catalog.nessie.warehouse: "s3://kyt-lake/warehouse"
  spark.sql.extensions: "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
```

---

## 7. K-Score (Anomaly Detection)

**Файл:** `src/kyt_engine/models/kscore.py`

K-Score — это unsupervised детектор аномалий, основанный на статистическом отклонении признаков текущей транзакции от baseline-распределения адреса. В отличие от supervised LightGBM, K-Score не требует меток и работает в реальном времени.

### 7.1 Baseline-окна

Для каждого адреса вычисляется baseline-распределение по 191 признаку на первых 6 временных шагах (~$t \leq 6$):

$$\mu_{addr,f} = \frac{1}{|W|}\sum_{x \in W_{addr}} x_f, \quad \sigma_{addr,f} = \sqrt{\frac{1}{|W|}\sum_{x \in W_{addr}} (x_f - \mu_{addr,f})^2}$$

где $W_{addr}$ — множество транзакций адреса в baseline-окне, $f$ — индекс признака.

### 7.2 Z-Score

Для каждой новой транзакции $x$ по каждому признаку $f$ вычисляется z-score:

$$z_f(x) = \frac{x_f - \mu_{addr,f}}{\sigma_{addr,f} + \epsilon}$$

$\epsilon = 10^{-8}$ предотвращает деление на ноль для адресов с константным baseline-признаком.

### 7.3 K-Score агрегация

K-Score = среднее абсолютных z-score по 191 признаку, нормированное в [0, 1]:

$$k(x) = \min\left(\frac{1}{191}\sum_{f=1}^{191} |z_f(x)|, \; 1.0\right)$$

Нормализация выполняется через эмпирический 99-й перцентиль, вычисленный на обучающей выборке.

### 7.4 Зоны риска

| Зона | Диапазон | Интерпретация |
|------|----------|---------------|
| GREEN | $k < 0.3$ | Поведение соответствует baseline-окну |
| YELLOW | $0.3 \leq k \leq 0.7$ | Умеренное отклонение, требует внимания |
| RED | $k > 0.7$ | Сильное отклонение, вероятная аномалия |

### 7.5 Результаты на production-данных

| Метрика | Значение |
|---------|----------|
| K-Score mean | 0.162 |
| K-Score std | 0.084 |
| GREEN (< 0.3) | 41,434 транзакций (88.8%) |
| YELLOW (0.3–0.7) | 4,987 транзакций (10.7%) |
| RED (> 0.7) | 143 транзакций (0.3%) |
| Корреляция с label | 0.41 |

---

## 8. Triage System

**Файл:** `src/kyt_engine/models/triage.py`

Triage — это risk-based decision tree, преобразующий выходы Unified Scorer в один из трёх уровней обработки кейса: автоматическое закрытие, приоритетный анализ или немедленная эскалация.

### 8.1 Решающее дерево

```
IF    K-Score < 0.3 AND p_LGBM > 0.9   → AUTO_CLOSE
ELIF  K-Score > 0.7 OR  entropy < 0.3   → ESCALATION
ELSE                                    → PRIORITY
```

**Логика правил:**

- **AUTO_CLOSE:** низкая аномальность (K-Score < 0.3) при высокой уверенности модели (LGBM > 0.9) — типичный паттерн нормального адреса, маркированного как illicit исторически
- **ESCALATION:** либо экстремальный K-Score (> 0.7), либо низкая энтропия предсказания (< 0.3) — модель уверена и сигнал сильный, требует срочной проверки
- **PRIORITY:** все остальные случаи — дефолтный режим, попадает в очередь аналитика

### 8.2 Распределение на production-данных

| Уровень | Процент | Описание |
|-----------|---------|----------|
| AUTO_CLOSE | 0.0% | low risk, высокая уверенность |
| PRIORITY | 99.7% | средний риск, нужен анализ |
| ESCALATION | 0.3% | высокий риск, срочная проверка |

Дисбаланс в сторону PRIORITY отражает консервативную политику AML: только явные случаи автоматизируются, остальное передаётся людям.

### 8.3 Метрики Triage

Triage минимизирует ручную нагрузку на аналитиков, отфильтровывая заведомо чистые случаи (AUTO_CLOSE) и приоритизируя опасные (ESCALATION). В production метрики отслеживаются через Prometheus:

```
kyt_alerts_total{triage="AUTO_CLOSE|PRIORITY|ESCALATION"}
```

---

## 9. Unified Scorer

**Файл:** `src/kyt_engine/models/unified_scorer.py`

Unified Scorer — production-компонент, объединяющий четыре сигнала (LightGBM, K-Score, VAE, External Labels) во взвешенный `risk_score ∈ [0, 1]` и присваивающий `risk_zone` и `triage_level`.

### 9.1 Формула ансамбля

$$P_{\text{risk}}(x) = 0.50 \cdot p_{\text{LGBM}}(x) + 0.20 \cdot k(x) + 0.15 \cdot \hat{s}_{\text{VAE}}(x) + 0.15 \cdot r_{\text{ext}}(x)$$

| Компонент | Вес | Источник |
|-----------|-----|----------|
| LightGBM | 0.50 | $p_{\text{LGBM}}(x) \in [0,1]$ — supervised probability |
| K-Score | 0.20 | $k(x) \in [0,1]$ — unsupervised anomaly magnitude |
| VAE | 0.15 | $\hat{s}_{\text{VAE}}(x) \in [0,1]$ — reconstruction anomaly |
| External | 0.15 | $r_{\text{ext}}(x) \in [0,1]$ — risk intelligence |

### 9.2 Маппинг в risk_zone

| Зона | Диапазон $P_{\text{risk}}$ | Действие |
|------|-----------------------------|----------|
| GREEN | $< 0.3$ | Обычная обработка, не подозрительная |
| YELLOW | $0.3 \leq P < 0.7$ | Помечена для мониторинга |
| RED | $\geq 0.7$ | Требует немедленного внимания |

### 9.3 Маппинг в triage_level

`triage_level` вычисляется отдельным decision tree (см. §8) и принимает значения: `AUTO_CLOSE`, `PRIORITY`, `ESCALATION`.

### 9.4 API-контракт

**Вход (`ScoringRequest`):**

```python
{
  "tx_id": str,
  "from_address": str,
  "to_address": str,
  "value": float,
  "gas_price": float,
  "gas_used": int,
  "timestamp": int,
  "features": dict  # опционально f0-f164
}
```

**Выход (`ScoringResponse`):**

```python
{
  "tx_id": str,
  "risk_score": float,        # ∈ [0, 1]
  "risk_zone": str,           # "GREEN" | "YELLOW" | "RED"
  "triage_level": str,        # "AUTO_CLOSE" | "PRIORITY" | "ESCALATION"
  "lgbm_proba": float,
  "k_score": float,
  "vae_anomaly": float,
  "external_risk": float,
  "top_reasons": list[dict]   # SHAP top-3
}
```

---

## 10. Active Learning

**Файл:** `src/kyt_engine/training/active_learning.py`

Active Learning реализует human-in-the-loop стратегию маркировки для улучшения модели без полного переобучения. Аналитики маркируют только информативные сэмплы, выбранные через uncertainty sampling.

### 10.1 Стратегия приоритизации

Используется комбинация двух сигналов неопределённости: **энтропии предсказания** (supervised uncertainty) и **K-Score** (unsupervised anomaly).

```
HIGH:   entropy > 0.7 AND k_score > 0.5
MEDIUM: entropy > 0.7 OR  k_score > 0.5
LOW:    иначе
```

**Энтропия предсказания** вычисляется как:

$$H(p) = -p \log_2 p - (1-p) \log_2 (1-p)$$

где $p = p_{\text{LGBM}}(x)$. Высокая энтропия (близко к 1) означает, что модель не уверена в классе.

**Комбинация с K-Score** позволяет находить сэмплы, которые одновременно (a) неоднозначны для supervised-модели и (b) статистически аномальны — это наиболее информативные точки для обучения.

### 10.2 Распределение приоритетов

| Приоритет | Количество | Доля | Действие |
|-----------|------------|------|----------|
| HIGH | 0 | 0.0% | Немедленная маркировка senior-аналитиком |
| MEDIUM | 145 | 29.0% | Маркировка в течение 24 часов |
| LOW | 355 | 71.0% | Бэклог, маркировка при наличии ресурсов |
| **Итого** | **500** | **100.0%** | — |

### 10.3 FeedbackLoop

После маркировки аналитиком сэмплы поступают в `FeedbackLoop.incremental_retrain(existing_model, new_labels_df)`:

```python
def incremental_retrain(existing_model, new_labels_df) -> dict:
    """
    Returns updated model with init_model=existing_model.
    Continues training on the augmented dataset without
    full retraining from scratch.
    """
```

Механизм `init_model=existing_model` (LightGBM `init_model` параметр) позволяет дообучать модель на новых данных с сохранением ранее выученных паттернов. Это критически важно для production, где полное переобучение занимает часы, а инкрементальное — минуты.

### 10.4 Цикл обратной связи

```
UncertaintySampler → SelectedSample.priority
       ↓
Analyst Labeling (Streamlit UI)
       ↓
FeedbackLoop.incremental_retrain()
       ↓
ModelRegistry (Iceberg) → новый snapshot
       ↓
ModelLoader hot-reload в FastAPI
```

---

## 11. External Labels (OFAC, GoPlus, ScamDB)

**Файл:** `src/kyt_engine/data/scraper.py`

Внешние источники риска дополняют supervised-сигнал, предоставляя ground-truth информацию об адресах из внешних баз.

### 11.1 Источники

| Источник | Тип | Частота обновления | Покрытие |
|----------|-----|---------------------|----------|
| OFAC SDN | Санкционные списки | Ежедневно | Глобальное, ~13k адресов |
| GoPlus Security | Токен-секьюрити | Real-time | EVM-токены, honeypot-детекция |
| ScamDB / Chainabuse | Fraud reports | Real-time | Мошеннические адреса по репортам |

### 11.2 ExternalLabelStore

Класс `ExternalLabelStore` обеспечивает:

- **Confidence scoring:** каждый лейбл имеет оценку достоверности $\in [0, 1]$
- **Source attribution:** указание источника для аудита и регуляторных требований
- **TTL-кэширование:** 24 часа, чтобы избежать повторных API-вызовов
- **Batch-обновления:** инкрементальная индексация новых лейблов
- **Iceberg-схема:** `address (PK), label, source, confidence, timestamp, metadata (JSON)`

### 11.3 Интеграция с Unified Scorer

External risk вычисляется как максимум confidence по всем источникам для данного адреса:

$$r_{\text{ext}}(x) = \max_{\text{source}} \text{confidence}_{\text{source}}(\text{from}(x)) \cup \text{confidence}_{\text{source}}(\text{to}(x))$$

Если адрес не найден ни в одном источнике, $r_{\text{ext}}(x) = 0$.

---

## 12. Kafka + Flink Streaming Ingestion

**Файлы:** `src/kyt_engine/ingestion/kafka_producer.py`, `flink_job.py`

Real-time pipeline обеспечивает приём блокчейн-транзакций с минимальной задержкой и exactly-once гарантиями.

### 12.1 Архитектура потока

```
RPC Node → Avro encode → Kafka (raw_txs) → Flink SQL → Iceberg (features)
```

### 12.2 Avro-схема транзакции

```json
{
  "type": "record",
  "name": "RawTransaction",
  "fields": [
    {"name": "tx_id",         "type": "string"},
    {"name": "block_height",  "type": "long"},
    {"name": "timestamp",     "type": "long"},
    {"name": "from_address",  "type": "string"},
    {"name": "to_address",    "type": "string"},
    {"name": "value",         "type": "double"},
    {"name": "gas_price",     "type": "double"},
    {"name": "gas_used",      "type": "long"},
    {"name": "input_data",    "type": "bytes"},
    {"name": "ingestion_ts",  "type": "long"}
  ]
}
```

### 12.3 Flink-конфигурация

- **Checkpointing:** ровно каждые 60 секунд, RocksDB state backend
- **Watermark strategy:** по `ingestion_ts` с допуском out-of-orderness 5 секунд
- **Window aggregation:** tumbling windows по адресам, 30-дневный горизонт
- **Sink:** Iceberg `features` table, partitioned by `days(timestamp)`

### 12.4 Exactly-once гарантии

Flink использует two-phase commit (2PC) с Kafka как источник и Iceberg как приёмник. Это гарантирует, что каждая транзакция записывается в `features` ровно один раз, даже при сбоях.

---

## 13. Iceberg Model Registry

**Файл:** `src/kyt_engine/data/iceberg_store.py`

Iceberg-таблица `models` обеспечивает версионированное хранение моделей с полной аудит-трассой.

### 13.1 Схема

```
model_id (PK): string
model_type: string         // "lightgbm" | "vae" | "ensemble"
version: string
metrics: string            // JSON: {auc_roc, f1, precision, recall, ...}
artifact_path: string      // S3 path к .pkl файлу
trained_at: timestamp
training_data_snapshot: string  // Iceberg snapshot-ID features table
metadata: string           // JSON: гиперпараметры, конфигурация
```

### 13.2 Возможности

- **Time-travel queries:** загрузка модели по `training_data_snapshot` для воспроизводимости экспериментов
- **Schema evolution:** добавление новых метрик (например, `calibration_error`) без миграции
- **ACID-транзакции:** атомарный promote в production — невозможно случайно подключить половину новой версии
- **Side-by-side comparison:** параллельный запуск нескольких версий через `ModelLoader(version=...)`
- **Snapshot lineage:** каждая запись `predictions` ссылается на конкретный `model_id`, обеспечивая полный аудит

### 13.3 Интеграция с MLflow

MLflow используется для трекинга экспериментов (гиперпараметры, метрики, артефакты), а Iceberg — для production-реестра. Двухуровневая архитектура:

```
MLflow Tracking (experiments) ── promote ──→ Iceberg models (production)
```

При promote в Iceberg автоматически публикуется webhook для Kubernetes-деплоймента, который обновляет ConfigMap `MODEL_VERSION` и триггерит rolling restart FastAPI-подов.

---

## 14. Конвейер обучения

### 14.1 Обоснование временной валидации

Стандартное стратифицированное разбиение 80/20 (`random_state=42`) используется как основная стратегия валидации. Однако для промышленного развёртывания рекомендуется временная валидация из-за события закрытия Dark Market на временном шаге 45 [15]. Временное разбиение обучает на временных шагах 1–44 и оценивает на шагах 45–49, имитируя реальное развёртывание, где модели должны обобщаться на ненаблюдённые будущие периоды.

### 14.2 Процедура псевдо-разметки

```
Вход: Базовая модель M, признаки X, метки y, порог достоверности τ = 0.95

1. Предсказание вероятностей для неразмеченных сэмплов:
   p = M.predict_proba(X_unknown)[:, 1]

2. Назначение псевдо-меток:
   y_pseudo[i] = 1 если p[i] ≥ τ        (высокодостоверная нелегитимная)
   y_pseudo[i] = 0 если p[i] ≤ (1 - τ)  (высокодостоверная легитимная)
   отбросить    если (1 - τ) < p[i] < τ  (неоднозначная)

3. Расширение обучающего набора:
   X_aug = concat(X_known, X_pseudo)
   y_aug = concat(y_known, y_pseudo)

4. Переобучение M на (X_aug, y_aug)
```

### 14.3 Генерация эмбеддингов Node2Vec

Когда требуются графовые эмбеддинги, конвейер выполняет:

1. **Построение графа:** Построение неориентированного графа NetworkX из списка рёбер (234,355 рёбер)
2. **Случайные блуждания:** 10 блужданий длиной 40 на узел, с $p=1.0$, $q=1.0$ (баланс BFS-DFS)
3. **Обучение Skip-Gram:** 5 эпох, окно=10, 5 негативных сэмплов, $\alpha=0.025$
4. **Извлечение эмбеддингов:** 64-мерные векторы на узел

Отсутствующие узлы (не в графе) получают нулевые векторы.

---

## 15. Сохранение моделей

Обученные модели сериализуются как pickle-файлы в директории `models/` и регистрируются в Iceberg `models` table:

```
models/
├── lightgbm.pkl          # Экземпляр LightGBMClassifier
├── autoencoder.pkl       # Экземпляр AutoencoderDetector
├── ensemble.pkl          # Экземпляр StackingEnsemble
├── kscore.pkl            # Экземпляр KScoreDetector
├── triage.pkl            # Параметры decision tree
├── unified_scorer.pkl    # Ансамбль с весами
└── top20_feature_importance.csv  # Рейтинг важности признаков
```

Все модели поддерживают `predict_proba(X)`, возвращающий массив $(n, 2)$ с $[P(\text{legit}), P(\text{illicit})]$, и `predict(X)`, возвращающий бинарные метки. KScore возвращает скалярный anomaly score ∈ [0, 1], UnifiedScorer возвращает `ScoringResponse` (см. §9.4).
