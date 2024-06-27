from dash import Dash, dcc, html, Input, Output
import plotly.express as px
# from datetime import datetime, timedelta
import holidays
import dateutil

import pandas as pd

nox_data = pd.read_csv('./data/NO2_mean_df_2019_2021.csv')
nox_data.rename(columns={'ADM2_NAME': 'City', "ADM1_NAME": 'State',
                         'DATE': 'Date'},
                inplace=True)
nox_data['Date'] = pd.to_datetime(nox_data['Date'])

app = Dash(__name__)


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

    dcc.Graph(id='time-series')
],

    style={'background-color': 'rgb(119,136,153)'})


@app.callback(
    Output('time-series', 'figure'),
    [Input('city-dd', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_county(selected_cities, start_date, end_date):
    # Convert the dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if selected_cities:
        filtered_data = nox_data[nox_data.City.isin(selected_cities)]
    else:
        filtered_data = nox_data.copy(deep=True)
    if start_date:
        filtered_data = filtered_data[filtered_data.Date > start_date]
    if end_date:
        filtered_data = filtered_data[filtered_data.Date < end_date]

    figure = px.line(filtered_data, x='Date', y='mean',
                     hover_data=['State', 'City', 'Date', 'mean', 'DOW', 'DOY'],
                     color='City', markers=True)
    figure.update_layout({'paper_bgcolor': 'rgb(44,44,44)', 'font': {'color': 'white'}})
    # If I had the population data here,
    #    I could assign size='population'

    # Add US holidays
    years = range(start_date.year, end_date.year + 1)
    us_holidays = holidays.US(years=years) + get_custom_holidays(years)
    holidays_in_range = [(pd.Timestamp(date), name) for date, name in us_holidays.items() if
                         start_date <= pd.Timestamp(date) <= end_date]
    for holiday_date, holiday_name in holidays_in_range:
        figure.add_vline(x=holiday_date, line=dict(color='red', dash='dash', width=1))
        figure.add_annotation(
            x=holiday_date,
            y=1,
            yref='paper',
            showarrow=False,
            text=holiday_name,
            xanchor='left',
            textangle=-90,
            font=dict(color='red')
        )
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
