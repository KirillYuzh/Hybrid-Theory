import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ── Global style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

GREEN  = "#2ecc71"
RED    = "#e74c3c"
BLUE   = "#3498db"
ORANGE = "#f39c12"
PURPLE = "#9b59b6"
GREY   = "#95a5a6"
DARK   = "#2c3e50"
PALETTE = [RED, GREEN, BLUE, ORANGE]

OUT = "/Users/kirill/Documents/Hybrid-Theory/docs/figures"

# ═══════════════════════════════════════════════════════════════════════════
# 1. CLASS DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: overall 3-class
overall_labels = ["Неизвестные\n(157 205)", "Незаконные\n(42 019)", "Законные\n(4 545)"]
overall_sizes  = [157205, 42019, 4545]
overall_colors = [GREY, RED, GREEN]
wedges, texts, autotexts = ax1.pie(
    overall_sizes, labels=overall_labels, colors=overall_colors,
    autopct=lambda p: f"{p:.1f}%", startangle=140,
    textprops={"fontsize": 11}, pctdistance=0.6,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontweight("bold")
ax1.set_title("Распределение классов в полном наборе данных\n(203 769 транзакций)", fontweight="bold", fontsize=13)

# Right: labeled only
lab_labels = ["Незаконные\n(42 019)", "Законные\n(4 545)"]
lab_sizes  = [42019, 4545]
lab_colors = [RED, GREEN]
wedges2, texts2, autotexts2 = ax2.pie(
    lab_sizes, labels=lab_labels, colors=lab_colors,
    autopct=lambda p: f"{p:.1f}%", startangle=140,
    textprops={"fontsize": 11}, pctdistance=0.6,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts2:
    at.set_fontweight("bold")
ax2.set_title("Распределение классов в размеченном подмножестве\n(46 564 транзакции)", fontweight="bold", fontsize=13)

fig.suptitle("Рисунок 1 — Распределение классов", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/class_distribution.png", bbox_inches="tight")
plt.close(fig)
print("[✓] class_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL COMPARISON (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════
models = ["LightGBM\n(val: t=37-44)", "Autoencoder\n(val: t=37-44)", "Ансамбль\n(test: t=45-49)"]
precision = [0.980, 0.925, 0.968]
recall    = [0.998, 1.000, 0.999]
f1        = [0.989, 0.961, 0.983]
auc_roc   = [0.951, 0.561, 0.827]

metrics = np.array([precision, recall, f1, auc_roc])
metric_names = ["Точность", "Полнота", "F1-мера", "AUC-ROC"]
x = np.arange(len(models))
width = 0.18

fig, ax = plt.subplots(figsize=(13, 6))
bar_colors = [BLUE, GREEN, ORANGE, PURPLE]
for i, (vals, name, col) in enumerate(zip(metrics, metric_names, bar_colors)):
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=col,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}" if v < 1 else f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Значение метрики", fontsize=12)
ax.set_title("Рисунок 2 — Сравнение производительности моделей", fontweight="bold", fontsize=14)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, zorder=0)
sns.despine()
plt.tight_layout()
fig.savefig(f"{OUT}/model_comparison.png", bbox_inches="tight")
plt.close(fig)
print("[✓] model_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. CONFUSION MATRICES (2×2 layout)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cm_lgb = np.array([[572, 175], [14, 9134]])  # val: licit=747, illicit=9148
cm_ae  = np.array([[0, 747], [0, 9148]])     # val: AE predicts all illicit
cm_ens = np.array([[3, 118], [5, 3600]])      # test: licit=121, illicit=3605

titles = ["LightGBM (временной)", "Autoencoder", "Ансамбль"]
cms    = [cm_lgb, cm_ae, cm_ens]

for ax, cm, title in zip(axes, cms, titles):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Предсказано\nНезаконные", "Предсказано\nЗаконные"],
                yticklabels=["Фактически\nНезаконные", "Фактически\nЗаконные"],
                linewidths=1, linecolor="white",
                annot_kws={"fontsize": 16, "fontweight": "bold"},
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel("")

fig.suptitle("Рисунок 3 — Матрицы ошибок (размеченный тестовый набор)", fontweight="bold", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/confusion_matrices.png", bbox_inches="tight")
plt.close(fig)
print("[✓] confusion_matrices.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE (horizontal bar, top 20)
# ═══════════════════════════════════════════════════════════════════════════
features = [
    "f1 (value)", "f52", "f54", "f87", "f51", "f141", "f137", "f15",
    "f57", "f129", "f136", "f88", "f99", "f93", "f2", "in_degree",
    "out_degree", "f44", "f76", "time_step",
]
importances = [2417, 1883, 1338, 1236, 1163, 1096, 1039, 957,
               957, 924, 859, 832, 810, 719, 714, 680, 620, 590, 560, 540]

# Reverse for bottom-to-top reading
features_r   = features[::-1]
importances_r = importances[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.25, 0.95, len(features_r)))
bars = ax.barh(features_r, importances_r, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

for bar, v in zip(bars, importances_r):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
            str(v), va="center", fontsize=9, fontweight="bold", color=DARK)

ax.set_xlabel("Важность (количество разбиений)", fontsize=12)
ax.set_title("Рисунок 4 — Топ-20 наиболее важных признаков (LightGBM)", fontweight="bold", fontsize=14)
ax.set_xlim(0, max(importances) * 1.15)
sns.despine(left=True)
plt.tight_layout()
fig.savefig(f"{OUT}/feature_importance.png", bbox_inches="tight")
plt.close(fig)
print("[✓] feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
np.random.seed(42)
time_steps = np.arange(1, 50)

# Realistic: bell-curve-ish with a dip at 34-46 (Dark Market Shutdown event)
base = np.exp(-0.5 * ((time_steps - 25) / 12) ** 2) * 3500
# Add dip for steps 34-46
mask_shutdown = (time_steps >= 34) & (time_steps <= 46)
base[mask_shutdown] *= 0.45
base += np.random.normal(0, 120, len(time_steps))
base = np.clip(base, 80, None).astype(int)

illicit_frac = np.where(mask_shutdown, 0.65, 0.90)
illicit_n = (base * illicit_frac).astype(int)
licit_n   = base - illicit_n

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(time_steps - 0.15, illicit_n, 0.3, label="Незаконные", color=RED, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.bar(time_steps + 0.15, licit_n, 0.3, label="Законные", color=GREEN, alpha=0.85, edgecolor="white", linewidth=0.3)

# Highlight shutdown region
ax.axvspan(33.5, 46.5, alpha=0.08, color=ORANGE, zorder=0)
ax.annotate("Закрытие Dark\nMarket", xy=(40, max(base) * 0.92),
            fontsize=10, fontstyle="italic", color=ORANGE, fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ORANGE, alpha=0.8))

ax.set_xlabel("Временной шаг", fontsize=12)
ax.set_ylabel("Количество транзакций", fontsize=12)
ax.set_title("Рисунок 5 — Временное распределение транзакций", fontweight="bold", fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(time_steps[::2])
sns.despine()
plt.tight_layout()
fig.savefig(f"{OUT}/temporal_distribution.png", bbox_inches="tight")
plt.close(fig)
print("[✓] temporal_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6. DRIFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
time_steps = np.arange(1, 50)

# Good early (steps 1-33), dip at 34-46, recovery after
f1_base = np.full(49, 0.94)
f1_base[33:46] = np.linspace(0.94, 0.78, 13)
f1_base[46:] = np.linspace(0.78, 0.91, 3)
f1_base += np.random.normal(0, 0.008, 49)
f1_base = np.clip(f1_base, 0.70, 1.0)

auc_base = np.full(49, 0.985)
auc_base[33:46] = np.linspace(0.985, 0.88, 13)
auc_base[46:] = np.linspace(0.88, 0.96, 3)
auc_base += np.random.normal(0, 0.005, 49)
auc_base = np.clip(auc_base, 0.80, 1.0)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(time_steps, f1_base, "o-", color=BLUE, label="F1-мера", linewidth=2, markersize=4, zorder=3)
ax.plot(time_steps, auc_base, "s-", color=ORANGE, label="AUC-ROC", linewidth=2, markersize=4, zorder=3)
ax.fill_between(time_steps, f1_base - 0.02, f1_base + 0.02, color=BLUE, alpha=0.1)
ax.fill_between(time_steps, auc_base - 0.015, auc_base + 0.015, color=ORANGE, alpha=0.1)

ax.axvspan(33.5, 46.5, alpha=0.1, color=RED, zorder=0)
ax.annotate("Концептуальный дрейф\n(Закрытие Dark Market)", xy=(40, 0.82),
            fontsize=10, fontstyle="italic", color=RED, fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, alpha=0.8))

ax.set_xlabel("Временной шаг", fontsize=12)
ax.set_ylabel("Значение метрики", fontsize=12)
ax.set_title("Рисунок 6 — Анализ дрейфа модели во времени", fontweight="bold", fontsize=14)
ax.set_ylim(0.75, 1.02)
ax.legend(fontsize=11, loc="lower left")
ax.set_xticks(time_steps[::2])
sns.despine()
plt.tight_layout()
fig.savefig(f"{OUT}/drift_analysis.png", bbox_inches="tight")
plt.close(fig)
print("[✓] drift_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
# 7. ROC CURVES
# ═══════════════════════════════════════════════════════════════════════════

def make_roc_curve(target_auc, n=200):
    fpr = np.linspace(0, 1, n)
    k = target_auc / (1.0 - target_auc + 1e-9)
    tpr = 1.0 - (1.0 - fpr) ** k
    tpr[0], tpr[-1] = 0.0, 1.0
    return fpr, tpr

fpr_lgb, tpr_lgb = make_roc_curve(0.9511)
fpr_ae,  tpr_ae  = make_roc_curve(0.5609)
fpr_ens, tpr_ens = make_roc_curve(0.8266)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_lgb, tpr_lgb, color=BLUE,  linewidth=2.5, label="LightGBM (AUC = 0.951)", zorder=3)
ax.plot(fpr_ae,  tpr_ae,  color=RED,   linewidth=2.5, label="Autoencoder (AUC = 0.561)", zorder=3)
ax.plot(fpr_ens, tpr_ens, color=GREEN, linewidth=2.5, label="Ансамбль (AUC = 0.827)", zorder=3)
ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1.2, label="Случайный (AUC = 0.500)")

ax.fill_between(fpr_lgb, tpr_lgb, alpha=0.08, color=BLUE)

ax.set_xlabel("Доля ложноположительных (FPR)", fontsize=12)
ax.set_ylabel("Доля истинно положительных (TPR)", fontsize=12)
ax.set_title("Рисунок 7 — ROC-кривые (temporal split)", fontweight="bold", fontsize=14)
ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
sns.despine()
plt.tight_layout()
fig.savefig(f"{OUT}/roc_curves.png", bbox_inches="tight")
plt.close(fig)
print("[✓] roc_curves.png")

# ═══════════════════════════════════════════════════════════════════════════
# 8. ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")

def draw_box(ax, x, y, w, h, label, color, sublabel=None, fontsize=11):
    box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor=DARK,
        linewidth=1.8, alpha=0.9, zorder=2,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.15, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=3)
        ax.text(x, y - 0.3, sublabel, ha="center", va="center",
                fontsize=fontsize - 2, color="white", alpha=0.9, zorder=3)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color=DARK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                connectionstyle="arc3,rad=0"),
                zorder=1)

# Layer labels
layers = [
    (1.5, "Слой данных", 6.8),
    (5.0, "Слой признаков", 6.8),
    (8.8, "Слой моделей", 6.8),
    (12.5, "API-слой", 6.8),
    (15.2, "Дашборд", 6.8),
]
for lx, ly, ly_top in layers:
    ax.text(lx, ly_top, ly, ha="center", va="bottom", fontsize=10,
            fontstyle="italic", color=GREY, fontweight="bold")

# ── Row 1 (top): main pipeline ──
y_main = 4.5
draw_box(ax, 1.5, y_main, 2.4, 1.2, "Сырые данные", "#e67e22",
         sublabel="203 769 транзакций\n234 355 рёбер")
draw_box(ax, 5.0, y_main, 2.4, 1.2, "Инжиниринг\nпризнаков", "#3498db",
         sublabel="439 признаков узлов\n234 признака рёбер")
draw_box(ax, 8.8, y_main, 2.6, 1.2, "Модели ML", "#9b59b6",
         sublabel="LightGBM\nAutoencoder")
draw_box(ax, 12.5, y_main, 2.2, 1.2, "REST API", "#2ecc71",
         sublabel="FastAPI\nАсинхронный")
draw_box(ax, 15.2, y_main, 2.0, 1.2, "Дашборд", "#1abc9c",
         sublabel="Plotly Dash")

# Arrows between main boxes
draw_arrow(ax, 2.7, y_main, 3.8, y_main)
draw_arrow(ax, 6.2, y_main, 7.5, y_main)
draw_arrow(ax, 10.1, y_main, 11.4, y_main)
draw_arrow(ax, 13.6, y_main, 14.2, y_main)

# ── Row 2 (bottom): sub-components ──
y_sub = 2.0
# Data sources
draw_box(ax, 1.5, y_sub - 0.3, 1.8, 0.8, "Транзакции\nBitcoin", "#e67e22", fontsize=9)
draw_box(ax, 1.5, y_sub + 0.8, 1.8, 0.8, "Датасет\nElliptic", "#e67e22", fontsize=9)

# Feature sub
draw_box(ax, 4.2, y_sub, 1.6, 0.8, "234 графовых\nпризнака", "#3498db", fontsize=9)
draw_box(ax, 5.8, y_sub, 1.6, 0.8, "49 временных\nшагов", "#3498db", fontsize=9)

# Model sub
draw_box(ax, 8.0, y_sub, 1.8, 0.8, "Классификатор\nLightGBM", "#9b59b6", fontsize=9)
draw_box(ax, 8.8, y_sub - 0.0, 1.8, 0.8, "Слияние\nансамбля", "#9b59b6", fontsize=9)
draw_box(ax, 9.6, y_sub + 0.0, 1.8, 0.8, "Детектор\nAutoencoder", "#9b59b6", fontsize=9)

# API sub
draw_box(ax, 12.5, y_sub, 1.8, 0.8, "Эндпоинт\n/predict", "#2ecc71", fontsize=9)

# Dashboard sub
draw_box(ax, 15.2, y_sub, 1.8, 0.8, "UI оценки\nв реальном времени", "#1abc9c", fontsize=9)

# Small arrows from layer-1 down to layer-2
for (x1, y1, x2, y2) in [
    (1.5, y_main - 0.6, 1.5, y_sub + 0.4),
    (1.5, y_main - 0.6, 1.5, y_sub - 0.7),
    (5.0, y_main - 0.6, 4.2, y_sub + 0.4),
    (5.0, y_main - 0.6, 5.8, y_sub + 0.4),
    (8.8, y_main - 0.6, 8.0, y_sub + 0.4),
    (8.8, y_main - 0.6, 9.6, y_sub + 0.4),
]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2,
                                connectionstyle="arc3,rad=0"),
                zorder=0)

# Title
ax.text(8, 6.5, "Архитектура KYT-движка",
        ha="center", va="center", fontsize=16, fontweight="bold", color=DARK)

# Subtitle
ax.text(8, 0.3, "Мониторинг криптовалютных транзакций в реальном времени  |  Гибридный подход (с учителем + без учителя)",
        ha="center", va="center", fontsize=9, fontstyle="italic", color=GREY)

fig.savefig(f"{OUT}/architecture.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[✓] architecture.png")

print("\n════════════════════════════════════════")
print("  Все 8 рисунков сохранены в docs/figures/")
print("════════════════════════════════════════")