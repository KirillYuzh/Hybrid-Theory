"""KYT Engine Executive Dashboard."""
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

app = dash.Dash(__name__, title="KYT Engine Dashboard")

# Mock data for demo (in production, loads from Iceberg/DuckDB)
np.random.seed(42)
n = 10000
mock_data = pd.DataFrame({
    'risk_score': np.random.beta(2, 5, n),
    'k_score': np.random.beta(2, 8, n),
    'zone': np.random.choice(['GREEN', 'YELLOW', 'RED'], n, p=[0.75, 0.20, 0.05]),
    'triage': np.random.choice(['auto_close', 'priority', 'escalation'], n, p=[0.65, 0.25, 0.10]),
    'time_step': np.random.randint(1, 50, n),
})

# Zone colors
zone_colors = {'GREEN': '#2ecc71', 'YELLOW': '#f39c12', 'RED': '#e74c3c'}

# Model performance comparison data
model_perf = pd.DataFrame({
    'model': ['LightGBM', 'VAE', 'AutoEncoder', 'Ensemble', 'GNN'],
    'precision': [0.92, 0.85, 0.83, 0.95, 0.88],
    'recall': [0.88, 0.91, 0.87, 0.93, 0.86],
    'f1': [0.90, 0.88, 0.85, 0.94, 0.87],
    'auc_roc': [0.96, 0.93, 0.91, 0.97, 0.94],
})

# External labels status
ext_labels = pd.DataFrame({
    'source': ['Elliptic', 'Chainalysis', 'OFAC', 'Local ML'],
    'total_labeled': [203769, 45000, 1200, 35000],
    'illicit_pct': [2.3, 5.1, 100.0, 3.8],
    'last_updated': ['2025-08-15', '2025-08-20', '2025-09-01', '2025-08-28'],
    'status': ['active', 'active', 'active', 'stale'],
})

app.layout = html.Div([
    html.H1("KYT Engine \u2014 \u0410\u043d\u0430\u043b\u0438\u0442\u0438\u0447\u0435\u0441\u043a\u0430\u044f \u043f\u0430\u043d\u0435\u043b\u044c", style={'textAlign': 'center', 'fontFamily': 'Arial'}),

    html.Div([
        # KPI Cards
        html.Div([
            html.Div([
                html.H3(f"{len(mock_data):,}"),
                html.P("\u0412\u0441\u0435\u0433\u043e \u0442\u0440\u0430\u043d\u0437\u0430\u043a\u0446\u0438\u0439"),
            ], className='kpi-card', style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center'}),
            html.Div([
                html.H3(f"{(mock_data['zone']=='RED').sum()}"),
                html.P("\u0412\u044b\u0441\u043e\u043a\u0438\u0439 \u0440\u0438\u0441\u043a (RED)"),
            ], className='kpi-card', style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center'}),
            html.Div([
                html.H3(f"{mock_data['risk_score'].mean():.3f}"),
                html.P("\u0421\u0440\u0435\u0434\u043d\u0438\u0439 Risk Score"),
            ], className='kpi-card', style={'backgroundColor': '#f39c12', 'color': 'white', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center'}),
            html.Div([
                html.H3(f"{(mock_data['triage']=='auto_close').mean()*100:.1f}%"),
                html.P("\u0410\u0432\u0442\u043e-\u0437\u0430\u043a\u0440\u044b\u0442\u0438\u0435"),
            ], className='kpi-card', style={'backgroundColor': '#2ecc71', 'color': 'white', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center'}),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '15px', 'marginBottom': '20px'}),

        # Charts row 1
        html.Div([
            dcc.Graph(
                id='risk-pie',
                figure=px.pie(
                    mock_data, names='zone', title='\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u0440\u0438\u0441\u043a\u043e\u0432',
                    color='zone', color_discrete_map=zone_colors
                ).update_traces(textinfo='percent+label'),
                style={'height': '350px'}
            ),
            dcc.Graph(
                id='kscore-hist',
                figure=px.histogram(
                    mock_data, x='k_score', color='zone', nbins=50,
                    title='\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 K-Score', color_discrete_map=zone_colors
                ),
                style={'height': '350px'}
            ),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'}),

        # Charts row 2
        html.Div([
            dcc.Graph(
                id='model-perf-bar',
                figure=go.Figure(data=[
                    go.Bar(name='Precision', x=model_perf['model'], y=model_perf['precision'], marker_color='#3498db'),
                    go.Bar(name='Recall', x=model_perf['model'], y=model_perf['recall'], marker_color='#2ecc71'),
                    go.Bar(name='F1', x=model_perf['model'], y=model_perf['f1'], marker_color='#f39c12'),
                    go.Bar(name='AUC-ROC', x=model_perf['model'], y=model_perf['auc_roc'], marker_color='#9b59b6'),
                ]).update_layout(
                    barmode='group',
                    title='\u0421\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u0435\u0439 (\u043c\u0435\u0442\u0440\u0438\u043a\u0438)',
                    yaxis_title='\u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435',
                ),
                style={'height': '350px'}
            ),
            dcc.Graph(
                id='temporal-drift',
                figure=px.line(
                    mock_data.groupby('time_step')['risk_score'].mean().reset_index(),
                    x='time_step', y='risk_score', title='\u0421\u0440\u0435\u0434\u043d\u0438\u0439 Risk Score \u043f\u043e \u0432\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u043c \u0448\u0430\u0433\u0430\u043c',
                    markers=True
                ).update_traces(line_color='#3498db'),
                style={'height': '350px'}
            ),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'}),

        # External labels table
        html.Div([
            html.H3("\u0421\u0442\u0430\u0442\u0443\u0441 \u0432\u043d\u0435\u0448\u043d\u0438\u0445 \u043c\u0435\u0442\u043e\u043a", style={'marginBottom': '10px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th('\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a'),
                    html.Th('\u0412\u0441\u0435\u0433\u043e \u043c\u0435\u0442\u043e\u043a'),
                    html.Th('% illicit'),
                    html.Th('\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u0435\u0435 \u043e\u0431\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435'),
                    html.Th('\u0421\u0442\u0430\u0442\u0443\u0441'),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(row['source']),
                        html.Td(f"{row['total_labeled']:,}"),
                        html.Td(f"{row['illicit_pct']}%"),
                        html.Td(row['last_updated']),
                        html.Td(
                            html.Span(row['status'], style={
                                'color': '#2ecc71' if row['status'] == 'active' else '#e74c3c',
                                'fontWeight': 'bold',
                            })
                        ),
                    ]) for _, row in ext_labels.iterrows()
                ]),
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': 'white', 'boxShadow': '0 1px 3px rgba(0,0,0,0.12)', 'borderRadius': '8px', 'overflow': 'hidden'}),
        ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto 20px auto'}),

    ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'}),
], style={'fontFamily': 'Arial', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
