import dash
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import detect_connect as dc
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, State, dash_table
import os
import json

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "DR"
server = app.server

# Function to get available dates
def get_available_dates():
    # Define the path to the data directory
    data_dir = 'data/apcn'

    # Get the list of dates
    dates = os.listdir(data_dir)

    # Remove any non-date files or directories
    dates = [date for date in dates if len(date) == 10 and date[4] == date[7] == '-']

    # Convert the dates to datetime objects
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

    # Get the earliest and latest dates
    min_date = min(dates)
    max_date = max(dates)

    return min_date, max_date

# Get the available dates
min_date, max_date = get_available_dates()

# Create the layout
app.layout = html.Div([
    dbc.Row(
        dbc.Col(html.H2("Dead Devices Detect Dashboard"), width={'size': 12}), 
        justify='center', align='center', style={'margin-top': '10px', 'margin-left': '40px'}
    ),

    dbc.Row([
        dbc.Col([
            html.Label('Infrastructure'),
            dcc.Dropdown(
                id='infrastructure-dropdown',
                options=[],
                value=[],
                multi=True)        
            ], width={'size': 5}),

        dbc.Col([
            html.Label('Date', style={'text-size': '8px', 'margin-right': '10px'}), 

            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                initial_visible_month=max_date,
                date=max_date,
                clearable=True,
                with_portal=True,
            )
        ], width={'size': 3}, align='center', style={'margin-left': '20px', 'margin-top': '15px'}),
        dbc.Col([
            html.Div(
                children=[html.Label('Missing Time Points Tolerance', style={'text-size': '8px', 'float': 'left'}),html.Abbr("\u2753", title="Choose the amount of missing time points to be considered as a dead device. There are 288 time points in a day.")
                          ]),
            dcc.Input(id='n-input', type='number', value=30)
        ], width={'size': 3}),
    ], justify='center', align='center', style={'margin-left': '10px'}, className="p-2 ms-3 me-3 bg-light border"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='subinterfaces-chart', figure={})
        ], width={'size': 9, 'offset': 1})
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(
                children=[html.Label('Continuous Time Range of Missing Time Points', style={'text-size': '8px', 'float': 'left'}),html.Abbr("\u2753", title="Choose the time range of missing time points to be considered as a dead device. Selected time range is continuous time points with no data received.")
                          ]),
            
            dcc.RangeSlider(
                id='time-range-slider', min=1200, max=1439, step=15, value=[1380, 1439],
                marks={
                    1200: {'label': '20:00'}, 1230: {'label': '20:30'},
                    1260: {'label': '21:00'}, 1290: {'label': '21:30'},
                    1320: {'label': '22:00'}, 1350: {'label': '22:30'},
                    1380: {'label': '23:00'}, 1410: {'label': '23:30'},
                    1439: {'label': '23:59'}
                })
        ], width={'size': 12}),
    ], justify='center', align='center', className="w-75 p-2 ms-5 bg-light border"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='missing-interfaces-chart', figure={})
        ], width={'size': 9, 'offset': 1})
    ]),

    html.Div(id='unmatched-interfaces-data', style={'display': 'none'}),

    dbc.Row([
        dbc.Col([
            html.Label('Completely Missing Routers/Interfaces', style={'text-size': '8px', 'margin-bottom': '10px'}),
            dash_table.DataTable(
                id='unmatched-interfaces-table',
                columns=[],
                data=[],  # Start with no data
                sort_action='native',  # Enable sorting
                page_size=15,  # Set the number of rows per page to 20
                filter_action='native',  # Enable column-wise filtering
                style_cell={'textAlign': 'left'},
            ), 
            html.Br(),
            html.Button('Export CSV', id='export-button', n_clicks=0)
        ], width={'size': 8, 'offset': 2})
    ])

])

@app.callback(
    [Output('subinterfaces-chart', 'figure'),
     Output('missing-interfaces-chart', 'figure')],
    [Input('infrastructure-dropdown', 'value'),
     Input('date-picker', 'date'),
     Input('time-range-slider', 'value'),
     Input('n-input', 'value')]
)
def update_data(infrastructure, date, time_range, n):

    start_minutes = time_range[0]
    end_minutes = time_range[1]

    # Split the date string and take only the necessary part
    date_parts = date.split("T")
    date_string = date_parts[0]

    # Convert the date string to a datetime object
    date = datetime.strptime(date_string, "%Y-%m-%d")

    # Convert the datetime object back to a string in the format "YYYY-MM-DD"
    date_string = date.strftime("%Y-%m-%d")
    # Extract the year, month, and day
    year = date.year
    month = date.month
    day = date.day

    # Read the new data
    df_dayin, df_input, df_dayout, df_output, df_couples = dc.read_data(date_string)  # Pass date_string as an argument

    # Set header
    dc.set_header(df_couples, ['id', 'router', 'interface', 'SAMPLES', 'infrastructure'])
    dc.set_header(df_dayin, ['id', 'router', 'interface', 'DATE', 'MAXvalue', 'AVERAGE', 'PERCENTILE', 'CROSS', 'RATIOMAX',
                             'RATIOAVERAGE', 'RATIOPERCENTILE', 'SAMPLES', 'infrastructure'])
    dc.set_header(df_dayout, ['id', 'router', 'interface', 'DATE', 'MAXvalue', 'AVERAGE', 'PERCENTILE', 'CROSS', 'RATIOMAX',
                             'RATIOAVERAGE', 'RATIOPERCENTILE', 'SAMPLES', 'infrastructure'])
    dc.set_header(df_input, ['id', 'router', 'interface', 'timestamp', 'value'])
    dc.set_header(df_output, ['id', 'router', 'interface', 'timestamp', 'value'])

    # Merge df_input with df_couples to get the 'infrastructure' information
    df_input_merged = pd.merge(df_input, df_couples, on=['id', 'router', 'interface'])

    # Filter the data based on the selected infrastructure
    if 'All' in infrastructure:
        # If 'All' is selected, include all infrastructures
        df_input_filtered = df_input_merged
    else:
        # Otherwise, filter based on the selected infrastructures
        df_input_filtered = df_input_merged[df_input_merged['infrastructure'].isin(infrastructure)]

    # Calculate start and end datetime objects based on the selected time range
    start_time = (datetime(year, month, day, 0, 0) + timedelta(minutes=start_minutes)).strftime('%H:%M')
    end_time = (datetime(year, month, day, 0, 0) + timedelta(minutes=end_minutes)).strftime('%H:%M')

    # Update the missing interfaces information
    subinterfaces_df = dc.get_dead_and_not_recovered_interfaces(df_input_filtered, df_couples, n, start_time, end_time)
    fullinterfaces_info = dc.get_subinterfaces_less_than_d_times(df_input_filtered, df_couples, n)

    # Define the missing interfaces figure
    subinterfaces_figure = dc.visualize_subinterfaces(df_input_filtered, fullinterfaces_info)
    subinterfaces_figure.update_layout(title='Routers/Interfaces with Missing Time Points')
    missing_interfaces_figure = dc.visualize_dead_and_not_recovered_interfaces(df_input_filtered, subinterfaces_df)
    missing_interfaces_figure.update_layout(title='Routers/Interfaces with Missing Time Points and No Recovery')


    return subinterfaces_figure, missing_interfaces_figure


@app.callback(
    Output('unmatched-interfaces-table', 'data'),
    [Input('date-picker', 'date')]
)
def update_table(date):
    # Split the date string and take only the necessary part
    date_parts = date.split("T")
    date_string = date_parts[0]

    # Convert the date string to a datetime object
    date = datetime.strptime(date_string, "%Y-%m-%d")

    # Convert the datetime object back to a string in the format "YYYY-MM-DD"
    date_string = date.strftime("%Y-%m-%d")

    # Read the new data
    df_dayin, df_input, df_dayout, df_output, df_couples = dc.read_data(date_string)

    # Set header
    dc.set_header(df_couples, ['id', 'router', 'interface', 'SAMPLES', 'infrastructure'])
    dc.set_header(df_input, ['id', 'router', 'interface', 'timestamp', 'value'])

    # Get the unmatched interfaces
    unmatched_df = dc.get_unmatched_interfaces(df_couples, df_input)
    
    return unmatched_df.to_dict('records')


@app.callback(
    Output('unmatched-interfaces-table', 'columns'),
    [Input('unmatched-interfaces-table', 'data')]
)
def update_table_columns(data):
    if data:
        return [{"name": i, "id": i} for i in data[0].keys()]
    else:
        return []

@app.callback(
    Output('infrastructure-dropdown', 'options'),
    [Input('date-picker', 'date')]
)
def update_infrastructure_options(date):
    # Split the date string and take only the necessary part
    date_parts = date.split("T")
    date_string = date_parts[0]

    # Convert the date string to a datetime object
    date = datetime.strptime(date_string, "%Y-%m-%d")

    # Convert the datetime object back to a string in the format "YYYY-MM-DD"
    date_string = date.strftime("%Y-%m-%d")

    # Read the new data
    df_dayin, df_input, df_dayout, df_output, df_couples = dc.read_data(date_string)

    # Set header
    dc.set_header(df_couples, ['id', 'router', 'interface', 'SAMPLES', 'infrastructure'])

    unique_infrastructures = dc.get_unique_infrastruct(df_couples)
    # Define the options for the infrastructure dropdown
    infrastructures = [{'label': i, 'value': i} for i in unique_infrastructures]

    return infrastructures

@app.callback(
    Output('export-button', 'children'),
    Input('export-button', 'n_clicks'),
    State('unmatched-interfaces-table', 'data'),
    prevent_initial_call=True
)
def export_csv(n_clicks, data):
    if n_clicks > 0:
        unmatched_df = pd.DataFrame(data)
        unmatched_df.to_csv('/tmp/unmatched_interfaces.csv', index=False)
        return 'CSV Exported'
    else:
        return 'Export CSV'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
