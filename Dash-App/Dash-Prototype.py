from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import pandas as pd

nox_data = pd.read_csv('./data/NO2_mean_df.csv')
nox_data.rename(columns={'ADM2_NAME': 'City', "ADM1_NAME":
                'State', 'DATE': 'Date'}, inplace=True)

app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(id='city-dd',
                 options=[
                     {'label': country, 'value': country} \
                     for country in nox_data['City'].unique()],
                 multi=True,
                 placeholder="Select a City",
                 style={'width': '500px', 'margin': '0 auto', 'background-color': 'rgb(192,192,192)'}),
    dcc.Graph(id='time-series')
    # TODO add a date selector
    ],
    style={'background-color': 'rgb(119,136,153)'})


@app.callback(
    Output('time-series', 'figure'),
    Input('city-dd', 'value')
)
def update_county(selected_cities):
    if selected_cities:
        filtered_data = nox_data[nox_data.City.isin(selected_cities)]
    else:
        filtered_data = nox_data.copy(deep=True)

    figure = px.line(filtered_data, x='Date', y='mean',
                     color='City', hover_data=['State', 'City', 'Date', 'mean', 'DOW', 'DOY'], markers=True)
    figure.update_layout({'paper_bgcolor': 'rgb(44,44,44)', 'font': {'color': 'white'}})
# If I had the population data here,
#    I could assign size='population'

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
