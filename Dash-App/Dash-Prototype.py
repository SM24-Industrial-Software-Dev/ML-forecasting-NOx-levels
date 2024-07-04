from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import holidays
import dateutil

import pandas as pd
from plotly.subplots import make_subplots

nox_data = pd.read_csv('./data/NO2_mean_df_2019_2021.csv')
nox_data.rename(columns={'ADM2_NAME': 'City', "ADM1_NAME": 'State',
                         'DATE': 'Date'},
                inplace=True)
nox_data['Date'] = pd.to_datetime(nox_data['Date'])

Monthly_nox = nox_data.copy(deep=True)
Monthly_nox.set_index('Date', inplace=True)
Monthly_nox = (Monthly_nox.groupby('City')['mean'].resample('ME').mean().reset_index())

app = Dash(__name__)

tab_selected_style = {
    'border': '2px solid #1915eb',
    'backgroundColor': 'black',
    'color': 'white',
    'padding': '6px'
}
tab_style = {
    'border': '1px solid grey',
    'backgroundColor': 'grey',
    'padding': '6px',
    'color': '#200031'
}
holiday_abbr = {
    'Independence Day': 'IND',
    'Independence Day (observed)': '',
    'Columbus Day': 'COL',
    'Veterans Day': 'VET',
    'Veterans Day (observed)': '',
    "New Year's Day": 'NYD',
    'Martin Luther King Jr. Day': 'MLK',
    "Washington's Birthday": 'PRS',
    'Memorial Day': 'MEM',
    'Labor Day': 'LAB',
    'Thanksgiving': 'THK',
    'Christmas Day': 'CHR',
    'Easter': 'eas'
}


def get_custom_holidays(years):
    custom_holidays = holidays.HolidayBase()
    for year in years:
        easter = dateutil.easter.easter(year)
        custom_holidays.append({
            easter: "Easter",
        })
    return custom_holidays


app.layout = html.Div([
    dcc.Dropdown(id='city-dd',
                 options=[
                     {'label': country, 'value': country} \
                     for country in nox_data['City'].unique()],
                 multi=True,
                 placeholder="Select a City",
                 style={'width': '500px', 'margin': '0 auto',
                        'background-color': 'rgb(192,192,192)'}),

    dcc.DatePickerRange(id='date-range',
                        start_date=nox_data['Date'].min(),
                        end_date=nox_data['Date'].max(),
                        style={'padding': '10px 450px', 'background-color': 'rgb(119,136,153)'}
                        ),
    dcc.Tabs(id="tabs-for-graph", value='hist', children=[
        dcc.Tab(label='Historical Data', value='hist', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Model', value='model', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Forecasts', value='forcast', style=tab_style, selected_style=tab_selected_style)
    ]),
    dcc.RadioItems(id='select-graph', value='daily', inline=True, options=[
        {'label': 'Daily', 'value': 'daily'},
        {'label': 'Monthly Average', 'value': 'monthly'},
        {'label': 'Yearly', 'value': 'yearly'}
    ]),
    dcc.Graph(id='time-series')
],

    style={'background-color': 'rgb(119,136,153)'})


def filter_data(unfiltered_data, selected_cities, start_date, end_date):
    if selected_cities:
        filtered_data = unfiltered_data[unfiltered_data.City.isin(selected_cities)]
    else:
        filtered_data = unfiltered_data.copy(deep=True)
    if start_date:
        filtered_data = filtered_data[filtered_data.Date > start_date]
    if end_date:
        filtered_data = filtered_data[filtered_data.Date < end_date]

    return filtered_data


@app.callback(
    [Output('time-series', 'figure'),
     Output('select-graph', 'style')],
    [Input('city-dd', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('select-graph', 'value'),
     Input('tabs-for-graph', 'value')]
)
def update_county(selected_cities, start_date, end_date, selected_view, selected_tab):
    if selected_tab == 'hist':
        # Convert the dates to datetime objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if selected_view == 'daily':
            # daily plot view
            filtered_data = filter_data(nox_data, selected_cities, start_date, end_date)
            figure = px.line(filtered_data, x='Date', y='mean',
                             hover_data=['State', 'City', 'Date', 'mean', 'DOW', 'DOY'],
                             color='City', markers=False, labels={'mean': 'NO2 concentration'},
                             title='tropospheric_NO2_column_number_density (mol/m^2)', height=450)
            # If I had the population data here,
            #    I could assign size='population'

            # Add US holidays
            years = range(start_date.year, end_date.year + 1)
            us_holidays = holidays.US(years=years) + get_custom_holidays(years)
            holidays_in_range = [(pd.Timestamp(date), name) for date, name in us_holidays.items()
                                 if start_date <= pd.Timestamp(date) <= end_date]
            for holiday_date, holiday_name in holidays_in_range:
                figure.add_vline(x=holiday_date, line=dict(color='red', dash='dash', width=1))
                figure.add_annotation(
                    x=holiday_date,
                    y=1, yref='paper',
                    showarrow=False, text=holiday_abbr[holiday_name],
                    xanchor='left', textangle=-90,
                    font=dict(color='red')
                )
        elif selected_view == 'monthly':
            # For the Monthly Plot
            filtered_data = filter_data(Monthly_nox, selected_cities, start_date, end_date)
            figure = px.line(filtered_data, x='Date', y='mean',
                             hover_data=['City', 'Date', 'mean'],
                             color='City', markers=True, labels={'mean': 'NO2 concentration'},
                             title='tropospheric_NO2_column_number_density (mol/m^2)', height=450)

            # Generate a list of dates that start new years
            year_start_dates = pd.date_range(start=start_date, end=end_date, freq='YS')

            # Add vertical lines at each new year
            for year_date in year_start_dates:
                figure.add_vline(x=year_date, line=dict(color='red', dash='dash', width=1))
                figure.add_annotation(
                    x=year_date,
                    y=1, yref='paper',
                    showarrow=False, text=str(year_date.year),
                    xanchor='left', textangle=-90,
                    font=dict(color='red')
                )
        else:
            # Year over year view
            nox_data['Year'] = pd.DatetimeIndex(nox_data['Date']).year
            nox_data['City-Year'] = nox_data['City'] + ' ' + nox_data['Year'].astype(str)
            filtered_data = filter_data(nox_data, selected_cities, start_date, end_date)
            figure = px.line(filtered_data, x='DOY', y='mean',
                             hover_data=['State', 'City', 'Date', 'mean', 'DOW', 'DOY', 'Year'],
                             color='City-Year', markers=False,
                             labels={'mean': 'NO2 concentration', 'DOY': 'Day of Year'},
                             title='tropospheric_NO2_column_number_density (mol/m^2)')
            selected_cities = selected_cities if selected_cities else nox_data['City'].unique()
            figure = make_subplots(rows=len(selected_cities), cols=1, shared_xaxes=True,
                                   subplot_titles=selected_cities)

            for i, city in enumerate(selected_cities):
                filtered_data = nox_data[nox_data['City'] == city]
                city_fig = px.line(filtered_data, x='DOY', y='mean',
                                   hover_data=['State', 'City', 'Date', 'mean', 'DOW', 'DOY', 'Year'],
                                   color='City-Year', markers=False,
                                   labels={'mean': 'NO2 concentration', 'DOY': 'Day of Year'})

                for trace in city_fig['data']:
                    figure.add_trace(trace, row=i + 1, col=1)
            figure.update_layout(height=400 * len(selected_cities))

        figure.update_layout({'paper_bgcolor': 'rgb(44,44,44)', 'font': {'color': 'white'},
                              'title': {'x': 0.45, 'xanchor': 'center'}}, yaxis={'title': None})
    else:
        figure = px.line()
    tabs = {'display': 'block' if selected_tab == 'hist' else 'none'}

    return figure, tabs


if __name__ == '__main__':
    app.run_server(debug=True)
