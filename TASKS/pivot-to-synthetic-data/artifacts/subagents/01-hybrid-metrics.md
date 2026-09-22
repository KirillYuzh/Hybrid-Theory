# 01 Subagent — Почему метрики Hybrid-Theory низкие

Отчёт subagent-1 (анализ Hybrid-Theory). Полный текст в сессии; здесь тезисы.

## Root-cause (приоритизированно)

### 1. Инверсия меток (главная причина «идеальных» цифр)
- `train_real.py:42` (git e1a3064, файл удалён из дерева): `classes["label"] = classes["class"].map({"1": 0, "2": 1})` — класс 1 (illicit) перевёрнут в 0.
- Реальное распределение: illicit 4 545 (2.2%), licit 42 019 (20.6%), unknown 157 205 (77.1%).
- Документация (`docs/results.md:13-19`, `docs/README.md:88-94`) докладывает ровно перевёрнутые числа («train: 32 910 illicit / 33 licit» — невозможно математически).
- LightGBM Precision 0.98 / Recall 0.99 / F1 0.99 — победа majority-класса после инверсии, а не детекция.

### 2. Метрики и графики в документах — синтетические/захардкоженные
- `docs/figures/generate_all.py`: классы `:35`, матрицы ошибок `:110-112`, temporal `:169-186` (np.random.seed(42), доли 0.65/0.90 выдуманы), drift `:212-227` (gauss-шум), ROC `:252-261` (параметрическая формула target_auc). Ни одна цифра не воспроизводится кодом.

### 3. Невалидный протокол оценки
- `train.py:133-135` — случайный сплит (не temporal).
- `train.py:144` + `lightgbm_model.py:83-93` — калибровка Isotonic и порог подбираются НА ОЦЕНОЧНОЙ ЖЕ выборке.

### 4. Утечка будущего в признаки
- `features/base.py:243-248` — in/out degree по всей истории адреса; `behavioral.py:227-234` — интервалы по всему trafic. Без временного окна → будущее встроено в обучающие признаки.

### 5. Дисбаланс + неверный выбор метрик
- 2.2% illicit → F1/recall некорректны; информативны AUC-PR, logloss, ECE, specificity.

### 6. Реальный датасет Elliptic в коде не используется
- `data/elliptic.py:11-13` читает синтетические nodes/edges/classes (1000 строк). `elliptic_txs_features.csv` (~690 МБ) нигде не открывается.

### 7. Слабые компоненты ансамбля (честно низкие метрики)
- K-Score: corr с label 0.41, RED recall 0.005 / F1 0.010.
- Autoencoder: AUC-ROC 0.56 при F1 0.96 (F1 = тривиальный артефакт majority: precision=0.9246 при «предсказать всех положительными»).
- Triage: 99.7% PRIORITY, 0% AUTO_CLOSE. Active Learning: HIGH-пул пуст.

### 8. Production-рассинхрон
- `app.py:30-44` грузит lightgbm*.pkl (10 фич f0..f9), а FeatureEngineer даёт 194 именованные фичи → нулевой вход в продакшене. UnifiedScorer = случайные матрицы np.random.randn.

### 9. Удалённые/убитые компоненты
- train_real.py, autoencoder.py, graph_embeddings.py, supplement.py удалены (git 9b56dc8); test_small.py импортирует несуществующие модули; моделей kscore/triage/unified_scorer нет.

## Главный вывод
«Плохие» метрики (VAE AUC 0.56, K-Score corr 0.41) — ЧЕСТНЫЕ сигналы. «Хорошие» (LightGBM 0.98, Ensemble 0.985) не воспроизводятся ни одним скриптом — артефакт инверсии меток + захардкоженных графиков.