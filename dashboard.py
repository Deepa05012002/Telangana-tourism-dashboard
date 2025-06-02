import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import webbrowser
import threading

# Load data
domestic_df = pd.read_excel("Telangana_Visitors_Domestic.xlsx")
foreign_df = pd.read_excel("Telangana_Visitors_Foreign.xlsx")
population_df = pd.read_excel("Telangana_Population_Districtwise.xlsx")

# Preprocess 
for df in [domestic_df, foreign_df]:
    df['Month'] = df['Month'].astype(str).str.zfill(2)
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'] + '-01')

# Helper functions
def group_data(df):
    return df.groupby('Date')['Visitors'].sum().reset_index()

def group_data_by_year(df):
    return df.groupby('Year')['Visitors'].sum().reset_index()

def hyderabad_monthly(df):
    return group_data(df[df['District'].str.lower() == 'hyderabad'])

def calculate_cagr(df, y1, y2):
    df_y1 = df[df['Year'] == y1].groupby('District')['Visitors'].sum()
    df_y2 = df[df['Year'] == y2].groupby('District')['Visitors'].sum()
    common = df_y1.index.intersection(df_y2.index)
    cagr_values = []
    for d in common:
        start, end = df_y1[d], df_y2[d]
        if start > 0 and end > 0:
            rate = (end / start) ** (1 / (y2 - y1)) - 1
        else:
            rate = 0
        cagr_values.append((d, rate))
    return pd.DataFrame(cagr_values, columns=['District', 'CAGR']).sort_values('CAGR', ascending=False).head(10)

def footfall_ratio(dom_df, for_df, pop_df):
    total_df = dom_df.copy()
    total_df['Visitors'] += for_df['Visitors']
    agg = total_df.groupby('District')['Visitors'].sum().reset_index()
    merged = pd.merge(agg, pop_df, on='District')
    merged['Footfall Ratio'] = merged['Visitors'] / merged['Population']
    return merged.sort_values('Footfall Ratio', ascending=False).head(10)

def forecast_sarima(df):
    try:
        data = group_data(df).set_index('Date')['Visitors']
        model = sm.tsa.statespace.SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12))
        results = model.fit(disp=False)
        steps = (2030 - data.index[-1].year) * 12 + (12 - data.index[-1].month + 1)
        forecast_result = results.get_forecast(steps=steps)
        forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
        pred_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        pred_mean.index = forecast_index
        conf_int.index = forecast_index
        conf_int.columns = ['Lower Bound', 'Upper Bound']
        return data, pred_mean, conf_int
    except Exception as e:
        print("Forecast Error:", e)
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()

def top10_visitors_by_district(dom_df, for_df, cat='domestic'):
    if cat == 'domestic':
        df = dom_df.groupby('District')['Visitors'].sum().reset_index()
    elif cat == 'foreign':
        df = for_df.groupby('District')['Visitors'].sum().reset_index()
    else:
        df = dom_df.copy()
        df['Visitors'] += for_df['Visitors']
        df = df.groupby('District')['Visitors'].sum().reset_index()
    return df.sort_values(by='Visitors', ascending=False).head(10)

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Telangana Tourism Dashboard"

def radio_card(title, id_, options, value):
    return dbc.Card([
        dbc.CardBody([
            html.H5(title, className="card-title text-light"),
            dbc.RadioItems(
                id=id_,
                options=[{'label': k.capitalize(), 'value': k} for k in options],
                value=value,
                inline=True,
                labelStyle={"color": "white", "margin-right": "15px"},
                inputClassName="me-2"
            )
        ])
    ], className="mb-3 bg-dark")

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("Telangana Tourism Dashboard", className="text-info fw-bold my-3"), width=12)]),
    dbc.Tabs([
        dbc.Tab(label="Monthly Visitors", children=[
            radio_card("Select Type:", 'monthly-toggle', ['domestic', 'foreign', 'both', 'total'], 'domestic'),
            dcc.Graph(id='monthly-graph')
        ]),
        dbc.Tab(label="Yearly Visitors", children=[
            radio_card("Select Type:", 'yearly-toggle', ['domestic', 'foreign', 'total'], 'domestic'),
            dcc.Graph(id='yearly-graph')
        ]),
        dbc.Tab(label="Hyderabad Monthly", children=[
            radio_card("Select Type:", 'hyd-toggle', ['domestic', 'foreign', 'both', 'total'], 'domestic'),
            dcc.Graph(id='hyd-graph')
        ]),
        dbc.Tab(label="Forecast", children=[
            radio_card("Select Forecast:", 'forecast-toggle', ['domestic', 'foreign', 'total'], 'domestic'),
            dcc.Graph(id='forecast-graph')
        ]),
        dbc.Tab(label="CAGR of Top 10 Districts", children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(sorted(domestic_df['Year'].unique()), id='cagr-year1', placeholder="Start Year")),
                dbc.Col(dcc.Dropdown(sorted(domestic_df['Year'].unique()), id='cagr-year2', placeholder="End Year")),
                dbc.Col(radio_card("CAGR Type", 'cagr-toggle', ['domestic', 'foreign', 'total'], 'domestic'))
            ], className='my-3'),
            html.Div(id='cagr-table')
        ]),
        dbc.Tab(label="Footfall Ratio of Top 10 Districts", children=[
            dcc.Graph(id='footfall-graph', figure=px.bar(
                footfall_ratio(domestic_df, foreign_df, population_df),
                x='District', y='Footfall Ratio',
                title="Top 10 Districts by Footfall Ratio",
                template='plotly_dark'
            ))
        ]),
        dbc.Tab(label="Top 10 Districtwise Visitors", children=[
            radio_card("Select Category", 'top10-toggle', ['domestic', 'foreign', 'total'], 'domestic'),
            html.Div(id='top10-table')
        ]),
    ])
], fluid=True)

# Callbacks
@app.callback(Output('monthly-graph', 'figure'), Input('monthly-toggle', 'value'))
def update_monthly(val):
    if val == 'both':
        d1, d2 = group_data(domestic_df), group_data(foreign_df)
        df = pd.merge(d1, d2, on='Date', suffixes=('_dom', '_for'))
        return px.line(df, x='Date', y=['Visitors_dom', 'Visitors_for'], 
                       title="Monthly Visitors (Domestic & Foreign)", template='plotly_dark')
    elif val == 'total':
        df = pd.merge(group_data(domestic_df), group_data(foreign_df), on='Date')
        df['Total'] = df['Visitors_x'] + df['Visitors_y']
        return px.bar(df, x='Date', y='Total', title="Total Monthly Visitors", template='plotly_dark')
    df = group_data(domestic_df if val == 'domestic' else foreign_df)
    return px.bar(df, x='Date', y='Visitors', title=f"{val.capitalize()} Monthly Visitors", template='plotly_dark')

@app.callback(Output('yearly-graph', 'figure'), Input('yearly-toggle', 'value'))
def update_yearly(val):
    if val == 'total':
        df1, df2 = group_data_by_year(domestic_df), group_data_by_year(foreign_df)
        df = pd.merge(df1, df2, on='Year')
        df['Total'] = df['Visitors_x'] + df['Visitors_y']
        return px.bar(df, x='Year', y='Total', title="Total Yearly Visitors", template='plotly_dark')
    df = group_data_by_year(domestic_df if val == 'domestic' else foreign_df)
    return px.bar(df, x='Year', y='Visitors', title=f"{val.capitalize()} Yearly Visitors", template='plotly_dark')

@app.callback(Output('hyd-graph', 'figure'), Input('hyd-toggle', 'value'))
def update_hyd(val):
    if val == 'both':
        d1, d2 = hyderabad_monthly(domestic_df), hyderabad_monthly(foreign_df)
        df = pd.merge(d1, d2, on='Date', suffixes=('_dom', '_for'))
        return px.line(df, x='Date', y=['Visitors_dom', 'Visitors_for'], 
                       title="Hyderabad Visitors (Domestic & Foreign)", template='plotly_dark')
    elif val == 'total':
        d1, d2 = hyderabad_monthly(domestic_df), hyderabad_monthly(foreign_df)
        df = pd.merge(d1, d2, on='Date')
        df['Total'] = df['Visitors_x'] + df['Visitors_y']
        return px.bar(df, x='Date', y='Total', title="Hyderabad Total Visitors", template='plotly_dark')
    df = hyderabad_monthly(domestic_df if val == 'domestic' else foreign_df)
    return px.bar(df, x='Date', y='Visitors', title=f"Hyderabad {val.capitalize()} Visitors", template='plotly_dark')

@app.callback(Output('forecast-graph', 'figure'), Input('forecast-toggle', 'value'))
def update_forecast(val):
    if val == 'total':
        df = domestic_df.copy()
        df['Visitors'] += foreign_df['Visitors']
    else:
        df = domestic_df if val == 'domestic' else foreign_df

    obs, pred, ci = forecast_sarima(df)
    fig = go.Figure()

    # Observed Data
    fig.add_trace(go.Scatter(
        x=obs.index, y=obs.values, mode='lines', name='Observed', line=dict(color='cyan')
    ))

    # Format the date as 'Mon YYYY' for hover
    forecast_dates = pred.index.strftime('%b %Y')
    if not ci.empty:
        hover_text = [
            f"<b>Month:</b> {date}<br>"
            f"<b>Forecast:</b> {fval:,.0f}<br>"
            f"<b>Lower Bound:</b> {lval:,.0f}<br>"
            f"<b>Upper Bound:</b> {uval:,.0f}"
            for date, fval, lval, uval in zip(forecast_dates, pred.values, ci['Lower Bound'], ci['Upper Bound'])
        ]
    else:
        hover_text = [
            f"<b>Month:</b> {date}<br><b>Forecast:</b> {fval:,.0f}"
            for date, fval in zip(forecast_dates, pred.values)
        ]

    # Forecast Line with custom hover
    fig.add_trace(go.Scatter(
        x=pred.index, y=pred.values, mode='lines', name='Forecast',
        line=dict(color='orange', dash='dot'),
        text=hover_text, hoverinfo='text'
    ))

    # Confidence Interval shaded area
    if not ci.empty:
        fig.add_trace(go.Scatter(
            x=ci.index.tolist() + ci.index[::-1].tolist(),
            y=ci['Upper Bound'].tolist() + ci['Lower Bound'][::-1].tolist(),
            fill='toself', fillcolor='rgba(255,140,0,0.3)', line=dict(width=0),
            name='Confidence Interval', hoverinfo='skip', showlegend=True
        ))

    fig.update_layout(
        title="Visitor Forecast with SARIMA Model (Till 2030)",
        xaxis_title="Date", yaxis_title="Number of Visitors",
        plot_bgcolor='#111111', paper_bgcolor='#111111',
        font=dict(color='white'), hovermode='x unified',
        legend=dict(orientation="h", y=-0.2)
    )

    return fig

@app.callback(Output('cagr-table', 'children'),
             [Input('cagr-year1', 'value'), Input('cagr-year2', 'value'), Input('cagr-toggle', 'value')])
def cagr_callback(y1, y2, typ):
    if not y1 or not y2 or y1 >= y2:
        return []

    df = domestic_df if typ == 'domestic' else foreign_df if typ == 'foreign' else domestic_df.copy()
    if typ == 'total':
        df['Visitors'] += foreign_df['Visitors']

    cagr_df = calculate_cagr(df, y1, y2)
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th("District"), html.Th("CAGR")])),
            html.Tbody([html.Tr([html.Td(row['District']), html.Td(f"{row['CAGR'] * 100:.2f}%")]) for _, row in cagr_df.iterrows()])
        ],
        bordered=True, hover=True, responsive=True, className="table table-light"
    )

@app.callback(Output('top10-table', 'children'), Input('top10-toggle', 'value'))
def update_top10_table(val):
    df = top10_visitors_by_district(domestic_df, foreign_df, val)
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th("District"), html.Th("Visitors")])),
            html.Tbody([html.Tr([html.Td(row['District']), html.Td(f"{row['Visitors']:,}")]) for _, row in df.iterrows()])
        ],
        bordered=True, hover=True, striped=True, responsive=True, className="table table-light"
    )

def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:8050/")

if __name__ == "__main__":
    # Start the browser shortly after the server launches
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
