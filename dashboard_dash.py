import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv('D://Research//madhavii//implementation//survey data.csv')
df['DENSITY 1 (total stock/Ha)'] = df['DENSITY 1 (total stock/Ha)'].replace('#REF!', np.nan)
df['DENSITY 1 (total stock/Ha)'] = pd.to_numeric(df['DENSITY 1 (total stock/Ha)'])

# Initialize the app
app = dash.Dash(__name__, title="Aquaculture Dashboard")

# Define colors
colors = {
    'background': '#f8f9fa',
    'text': '#212529',
    'primary': '#007bff',
    'success': '#28a745'
}

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("🐟 Aquaculture Production Dashboard", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': 30}),

        # Filters
        html.Div([
            html.Div([
                html.Label('Duration (Months)'),
                dcc.RangeSlider(
                    id='duration-slider',
                    min=df['DURATIONMONTHS'].min(),
                    max=df['DURATIONMONTHS'].max(),
                    value=[df['DURATIONMONTHS'].min(), df['DURATIONMONTHS'].max()],
                    marks={i: str(i) for i in range(int(df['DURATIONMONTHS'].min()), 
                                                     int(df['DURATIONMONTHS'].max())+1, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20px'}),

            html.Div([
                html.Label('Number of Pens'),
                dcc.Dropdown(
                    id='pens-dropdown',
                    options=[{'label': f'{i} Pen(s)', 'value': i} for i in sorted(df['PENS'].unique())],
                    value=sorted(df['PENS'].unique()),
                    multi=True
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20px'}),

            html.Div([
                html.Label('Area Range (Ha)'),
                dcc.RangeSlider(
                    id='area-slider',
                    min=df['Area (Ha 1)'].min(),
                    max=df['Area (Ha 1)'].max(),
                    value=[df['Area (Ha 1)'].min(), df['Area (Ha 1)'].max()],
                    marks={i: f'{i}' for i in np.arange(0, df['Area (Ha 1)'].max()+1, 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20px'})
        ], style={'marginBottom': 40}),

        # KPIs
        html.Div(id='kpi-cards', style={'marginBottom': 30}),

        # Charts Row 1
        html.Div([
            html.Div([dcc.Graph(id='production-histogram')], 
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='production-duration-scatter')], 
                     style={'width': '50%', 'display': 'inline-block'})
        ]),

        # Charts Row 2
        html.Div([
            html.Div([dcc.Graph(id='density-production-scatter')], 
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='pens-production-box')], 
                     style={'width': '50%', 'display': 'inline-block'})
        ]),

        # Charts Row 3
        html.Div([
            html.Div([dcc.Graph(id='vegetation-production-scatter')], 
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='correlation-heatmap')], 
                     style={'width': '50%', 'display': 'inline-block'})
        ])

    ], style={'padding': '20px', 'backgroundColor': colors['background']})
])

# Callbacks
@app.callback(
    [Output('kpi-cards', 'children'),
     Output('production-histogram', 'figure'),
     Output('production-duration-scatter', 'figure'),
     Output('density-production-scatter', 'figure'),
     Output('pens-production-box', 'figure'),
     Output('vegetation-production-scatter', 'figure'),
     Output('correlation-heatmap', 'figure')],
    [Input('duration-slider', 'value'),
     Input('pens-dropdown', 'value'),
     Input('area-slider', 'value')]
)
def update_dashboard(duration_range, pens_selected, area_range):
    # Filter data
    filtered_df = df[
        (df['DURATIONMONTHS'] >= duration_range[0]) &
        (df['DURATIONMONTHS'] <= duration_range[1]) &
        (df['PENS'].isin(pens_selected)) &
        (df['Area (Ha 1)'] >= area_range[0]) &
        (df['Area (Ha 1)'] <= area_range[1])
    ]

    # KPI Cards
    kpi_style = {
        'textAlign': 'center',
        'padding': '20px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
        'margin': '10px'
    }

    kpis = html.Div([
        html.Div([
            html.H4('Total Farms', style={'color': '#6c757d', 'fontSize': '14px'}),
            html.H2(f"{len(filtered_df)}", style={'color': colors['primary'], 'margin': '10px 0'})
        ], style={**kpi_style, 'width': '18%', 'display': 'inline-block'}),

        html.Div([
            html.H4('Avg Production', style={'color': '#6c757d', 'fontSize': '14px'}),
            html.H2(f"{filtered_df['PRODUCTION 1'].mean():.0f} kg", 
                   style={'color': colors['success'], 'margin': '10px 0'})
        ], style={**kpi_style, 'width': '18%', 'display': 'inline-block'}),

        html.Div([
            html.H4('Avg Production/Ha', style={'color': '#6c757d', 'fontSize': '14px'}),
            html.H2(f"{filtered_df['PRODUCTION 1/Ha'].mean():.0f} kg/Ha", 
                   style={'color': colors['success'], 'margin': '10px 0'})
        ], style={**kpi_style, 'width': '18%', 'display': 'inline-block'}),

        html.Div([
            html.H4('Avg Fish Weight', style={'color': '#6c757d', 'fontSize': '14px'}),
            html.H2(f"{filtered_df['AVG WEIGHT 1'].mean():.2f} kg", 
                   style={'color': colors['primary'], 'margin': '10px 0'})
        ], style={**kpi_style, 'width': '18%', 'display': 'inline-block'}),

        html.Div([
            html.H4('Avg Duration', style={'color': '#6c757d', 'fontSize': '14px'}),
            html.H2(f"{filtered_df['DURATIONMONTHS'].mean():.1f} mo", 
                   style={'color': colors['primary'], 'margin': '10px 0'})
        ], style={**kpi_style, 'width': '18%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-around'})

    # Chart 1: Production Histogram
    fig1 = px.histogram(filtered_df, x='PRODUCTION 1/Ha', nbins=20,
                        title="Production per Hectare Distribution")
    fig1.update_layout(showlegend=False)

    # Chart 2: Production vs Duration
    fig2 = px.scatter(filtered_df, x='DURATIONMONTHS', y='PRODUCTION 1',
                      size='Area (Ha 1)', color='AVG WEIGHT 1',
                      title="Production vs Duration",
                      color_continuous_scale='Viridis')

    # Chart 3: Density vs Production
    fig3 = px.scatter(filtered_df.dropna(subset=['DENSITY 1 (total stock/Ha)']),
                      x='DENSITY 1 (total stock/Ha)', y='PRODUCTION 1/Ha',
                      trendline="ols", title="Density vs Production/Ha")

    # Chart 4: Production by Pens
    fig4 = px.box(filtered_df, x='PENS', y='PRODUCTION 1/Ha',
                  title="Production by Number of Pens")

    # Chart 5: Vegetation vs Production
    fig5 = px.scatter(filtered_df, x='VEGETATION%', y='PRODUCTION 1/Ha',
                      color='AVERAGE_DEPTH/ft', size='Area (Ha 1)',
                      title="Vegetation vs Production")

    # Chart 6: Correlation Heatmap
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    fig6 = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                     title="Correlation Heatmap", color_continuous_scale='RdBu_r')

    return kpis, fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    # app.run_server(debug=True, port=8050)
    app.run(debug=True, port=8050)
    # app.run_server(debug=True, port=8050)
