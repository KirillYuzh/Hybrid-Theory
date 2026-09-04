"""Дашборд руководителя KYT Engine.

Отображает KPI: количество транзакций, распределение risk_score,
triage-уровни, рисковые зоны. Данные берутся из локального CSV-кэша
(создаётся при старте) и позволяют руководителю видеть текущее
состояние AML-конвейера без доступа к API.

Запуск:
    python -m kyt_engine.dashboard.app   # порт 8050
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html

# ---------------------------------------------------------------------------
# Утилита: генерация демо-данных для дашборда (когда нет продакшен-метрик)
# ---------------------------------------------------------------------------
def _ensure_data_cache() -> Path:
    """Создаёт каталог data/dashboard и возвращает путь."""
    cache_dir = Path(os.environ.get("KYT_DATA_DIR", "data")) / "dashboard"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_prediction_history() -> pd.DataFrame:
    """Загружает историю предсказаний из CSV или создаёт демо-историю.

    В production здесь читается таблица Iceberg ``predictions``.
    Для воспроизводимой демонстрации генерируем детерминированные данные.
    """
    cache = _ensure_data_cache()
    csv_path = cache / "predictions_history.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        # Оставляем последние 5000 записей, чтобы графики были читаемыми
        if len(df) > 5000:
            df = df.tail(5000)
        return df

    rng = np.random.default_rng(42)
    n = 2000
    now = pd.Timestamp.now().floor("min")

    timestamps = pd.date_range(end=now, periods=n, freq="5min")
    risk_score = np.clip(
        0.15 + 0.6 * np.abs(rng.normal(0, 1, n)) + 0.05 * rng.normal(0, 1, n),
        0.0,
        1.0,
    )
    lgbm_proba = np.clip(risk_score + rng.normal(0, 0.08, n), 0.0, 1.0)
    k_score = np.clip(np.abs(rng.normal(0.15, 0.25, n)), 0.0, 1.0)
    vae_anomaly = np.clip(np.abs(rng.normal(0.1, 0.2, n)), 0.0, 1.0)

    df = pd.DataFrame(
        {
            "tx_id": [f"tx_{i:06d}" for i in range(n)],
            "timestamp": timestamps,
            "risk_score": risk_score.round(6),
            "lgbm_proba": lgbm_proba.round(6),
            "k_score": k_score.round(6),
            "vae_anomaly": vae_anomaly.round(6),
        }
    )

    df["risk_zone"] = pd.cut(
        df["risk_score"],
        bins=[0.0, 0.3, 0.7, 1.0],
        labels=["GREEN", "YELLOW", "RED"],
    ).astype(str)

    # Triage level: эвристика по K-Score и risk_score
    def _triage(row: pd.Series) -> str:
        if row["risk_score"] < 0.2 and row["k_score"] < 0.3:
            return "AUTO_CLOSE"
        if row["risk_score"] > 0.7 or row["k_score"] > 0.7:
            return "ESCALATION"
        return "PRIORITY"

    df["triage_level"] = df.apply(_triage, axis=1)
    df["amount_usd"] = np.exp(rng.normal(6.0, 2.5, n)).round(2)

    df.to_csv(csv_path, index=False)
    return df


def _api_health() -> dict:
    """Проверяет доступность API KYT Engine."""
    import urllib.request

    api_host = os.environ.get("KYT_API_HOST", "kyt-api:8000")
    url = f"http://{api_host}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {"status": "unreachable", "models_loaded": False, "model_names": []}


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
df = _load_prediction_history()

# Исправляем возможные проблемы парсинга
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

# ---------------------------------------------------------------------------
# Dash-приложение
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    title="KYT Engine — Дашборд руководителя",
    update_title=None,
)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "24px", "background": "#f7f9fc"},
    children=[
        html.H1(
            "KYT Engine — панель руководителя",
            style={"color": "#1a2b4a", "marginBottom": "4px"},
        ),
        html.Div(
            id="api-status",
            style={"color": "#666", "marginBottom": "20px", "fontSize": "14px"},
        ),
        dcc.Interval(id="interval", interval=15_000, n_intervals=0),

        # KPI
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
            children=[
                html.Div(
                    id="kpi-total",
                    className="kpi-card",
                    style=kpi_style(),
                ),
                html.Div(
                    id="kpi-avg-risk",
                    className="kpi-card",
                    style=kpi_style(),
                ),
                html.Div(
                    id="kpi-escalations",
                    className="kpi-card",
                    style=kpi_style(),
                ),
                html.Div(
                    id="kpi-red",
                    className="kpi-card",
                    style=kpi_style(),
                ),
            ],
        ),

        # Основные графики
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
            children=[
                html.Div(
                    style={"flex": "1 1 48%", "background": "white", "borderRadius": "12px",
                           "padding": "16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"},
                    children=[
                        html.H3("Динамика risk_score", style={"marginTop": "0", "color": "#1a2b4a"}),
                        dcc.Graph(id="chart-risk-trend"),
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 48%", "background": "white", "borderRadius": "12px",
                           "padding": "16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"},
                    children=[
                        html.H3("Распределение по зонам", style={"marginTop": "0", "color": "#1a2b4a"}),
                        dcc.Graph(id="chart-zones"),
                    ],
                ),
            ],
        ),

        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
            children=[
                html.Div(
                    style={"flex": "1 1 48%", "background": "white", "borderRadius": "12px",
                           "padding": "16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"},
                    children=[
                        html.H3("Triage pipeline", style={"marginTop": "0", "color": "#1a2b4a"}),
                        dcc.Graph(id="chart-triage"),
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 48%", "background": "white", "borderRadius": "12px",
                           "padding": "16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"},
                    children=[
                        html.H3("Распределение K-Score", style={"marginTop": "0", "color": "#1a2b4a"}),
                        dcc.Graph(id="chart-kscore"),
                    ],
                ),
            ],
        ),

        # Таблица последних транзакций
        html.Div(
            style={"background": "white", "borderRadius": "12px", "padding": "16px",
                   "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"},
            children=[
                html.H3("Последние транзакции", style={"marginTop": "0", "color": "#1a2b4a"}),
                html.Div(id="table-recent", style={"overflowX": "auto"}),
            ],
        ),
    ],
)


def kpi_style() -> dict:
    return {
        "flex": "1 1 200px",
        "background": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
        "textAlign": "center",
    }


def _kpi_html(label: str, value: str, color: str = "#1a2b4a") -> html.Div:
    return html.Div(
        children=[
            html.Div(label, style={"fontSize": "13px", "color": "#888", "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": "700", "color": color}),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("api-status", "children"),
    Input("interval", "n_intervals"),
)
def update_api_status(_n: int) -> str:
    health = _api_health()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = health.get("status", "unknown")
    color = "#2e7d32" if status == "ok" else "#c62828"
    models = ", ".join(health.get("model_names", [])) if health.get("model_names") else "—"
    return html.Span(
        [
            f"API: ",
            html.Span(status, style={"color": color, "fontWeight": "bold"}),
            f" | Модели: {models} | Обновлено: {ts}",
        ]
    )


@callback(
    [
        Output("kpi-total", "children"),
        Output("kpi-avg-risk", "children"),
        Output("kpi-escalations", "children"),
        Output("kpi-red", "children"),
    ],
    Input("interval", "n_intervals"),
)
def update_kpi(_n: int) -> list[html.Div]:
    total = len(df)
    avg_risk = float(df["risk_score"].mean())
    esc = int((df["triage_level"] == "ESCALATION").sum())
    red = int((df["risk_zone"] == "RED").sum())
    return [
        _kpi_html("Всего транзакций", f"{total:,}".replace(",", " ")),
        _kpi_html("Средний риск", f"{avg_risk:.3f}", color="#1565c0"),
        _kpi_html("Эскалации", f"{esc:,}".replace(",", " "), color="#c62828"),
        _kpi_html("RED-зона", f"{red:,}".replace(",", " "), color="#c62828"),
    ]


@callback(
    Output("chart-risk-trend", "figure"),
    Input("interval", "n_intervals"),
)
def update_risk_trend(_n: int) -> dict:
    d = df.copy()
    d["hour_bucket"] = d["timestamp"].dt.floor("H")
    trend = d.groupby("hour_bucket")["risk_score"].agg(["mean", "max"]).reset_index()
    fig = px.line(
        trend,
        x="hour_bucket",
        y=["mean", "max"],
        labels={"value": "risk_score", "hour_bucket": "Время", "variable": "Метрика"},
        color_discrete_map={"mean": "#1565c0", "max": "#c62828"},
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320,
    )
    return fig


@callback(
    Output("chart-zones", "figure"),
    Input("interval", "n_intervals"),
)
def update_zones(_n: int) -> dict:
    counts = df["risk_zone"].value_counts().reindex(["GREEN", "YELLOW", "RED"], fill_value=0)
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map={"GREEN": "#2e7d32", "YELLOW": "#f9a825", "RED": "#c62828"},
        labels={"x": "Зона", "y": "Количество"},
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
        height=320,
    )
    return fig


@callback(
    Output("chart-triage", "figure"),
    Input("interval", "n_intervals"),
)
def update_triage(_n: int) -> dict:
    counts = df["triage_level"].value_counts().reindex(
        ["AUTO_CLOSE", "PRIORITY", "ESCALATION"], fill_value=0
    )
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color_discrete_map={"AUTO_CLOSE": "#2e7d32", "PRIORITY": "#f9a825", "ESCALATION": "#c62828"},
        hole=0.4,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=320)
    return fig


@callback(
    Output("chart-kscore", "figure"),
    Input("interval", "n_intervals"),
)
def update_kscore(_n: int) -> dict:
    fig = px.histogram(df, x="k_score", nbins=40, color_discrete_sequence=["#1565c0"])
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="K-Score",
        yaxis_title="Частота",
        height=320,
    )
    return fig


@callback(
    Output("table-recent", "children"),
    Input("interval", "n_intervals"),
)
def update_table(_n: int) -> html.Table:
    recent = df.tail(20).iloc[::-1]
    zone_color = {"GREEN": "#2e7d32", "YELLOW": "#f9a825", "RED": "#c62828"}

    rows = [
        html.Tr(
            [
                html.Th("tx_id", style=th_style()),
                html.Th("Время", style=th_style()),
                html.Th("Risk", style=th_style()),
                html.Th("LGBM", style=th_style()),
                html.Th("K-Score", style=th_style()),
                html.Th("Зона", style=th_style()),
                html.Th("Triage", style=th_style()),
                html.Th("Сумма, USD", style=th_style()),
            ]
        )
    ]
    for _, r in recent.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(r["tx_id"], style=td_style()),
                    html.Td(r["timestamp"].strftime("%m-%d %H:%M"), style=td_style()),
                    html.Td(f"{r['risk_score']:.3f}", style=td_style()),
                    html.Td(f"{r['lgbm_proba']:.3f}", style=td_style()),
                    html.Td(f"{r['k_score']:.3f}", style=td_style()),
                    html.Td(
                        html.Span(
                            r["risk_zone"],
                            style={
                                "background": zone_color.get(r["risk_zone"], "#999"),
                                "color": "white",
                                "padding": "2px 8px",
                                "borderRadius": "10px",
                                "fontSize": "12px",
                                "fontWeight": "bold",
                            },
                        ),
                        style=td_style(),
                    ),
                    html.Td(r["triage_level"], style=td_style()),
                    html.Td(f"{r['amount_usd']:,.0f}".replace(",", " "), style=td_style()),
                ]
            )
        )

    return html.Table(
        rows,
        style={"borderCollapse": "collapse", "width": "100%", "fontSize": "13px"},
    )


def th_style() -> dict:
    return {
        "padding": "8px 12px",
        "borderBottom": "2px solid #ddd",
        "textAlign": "left",
        "background": "#f5f7fa",
        "color": "#333",
    }


def td_style() -> dict:
    return {"padding": "8px 12px", "borderBottom": "1px solid #eee"}


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------
def main() -> None:
    port = int(os.environ.get("KYT_DASHBOARD_PORT", "8050"))
    host = os.environ.get("KYT_DASHBOARD_HOST", "0.0.0.0")
    print(f"KYT Dashboard listening on http://{host}:{port}")
    app.run(debug=False, host=host, port=port)


if __name__ == "__main__":
    main()