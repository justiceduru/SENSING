import datetime
import pandas as pd
import csv
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.externals import joblib

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

import numpy as np                               # vectors and matrices                 # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization
import csv
import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
scaler = StandardScaler()
from sklearn.externals import joblib

model = joblib.load("ML_Model3.pkl")
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators


api_key = 'UTBNYS0PG96GKLDJ'

# pip install pyorbital
from pyorbital.orbital import Orbital
satellite = Orbital('TERRA')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H2('Silicon Strategy (EBAY)'),
        html.H4('Use the following information to make intelligent decisions for technology stocks decide between a bear or a bull strategy today'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph', style={
            'height': 800
        },),
        dcc.Interval(
            id='interval-component',
            interval=5*60*1000, # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    ts = TimeSeries(key=api_key, output_format='pandas')
    ti = TechIndicators(key=api_key, output_format='pandas')
    period = 100
    GOOGLdata_ts, GOOGLmeta_data_ts = ts.get_intraday(symbol='EBAY', interval='5min', outputsize='compact')
    GOOGLdata_ti, GOOGLmeta_data_ti = ti.get_sma(symbol='EBAY', interval='5min', time_period=period, series_type='close')


    def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
        """
            series: pd.DataFrame
                dataframe with timeseries

            lag_start: int
                initial step back in time to slice target variable
                example - lag_start = 1 means that the model
                          will see yesterday's values to predict today

            lag_end: int
                final step back in time to slice target variable
                example - lag_end = 4 means that the model
                          will see up to 4 days back in time to predict today

            test_size: float
                size of the test dataset after train/test split as percentage of dataset

            target_encoding: boolean
                if True - add target averages to the dataset

        """

        # copy of the initial dataset
        data = pd.DataFrame(series.copy())
        data.columns = ["y"]

        # lags of series
        for i in range(lag_start, lag_end):
            data["lag_{}".format(i)] = data.y.shift(-i)

        # datetime features
        # train-test split
        y = data.dropna().y
        X = data.dropna().drop(['y'], axis=1)

        return X, y

    X, y = prepareData(GOOGLdata_ts['4. close'], lag_start=0, lag_end=19, test_size=0.3, target_encoding=True)

    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X)

    pred = model.predict(X_test_scaled)
    closep = GOOGLdata_ts.iloc[0]['4. close']
    smav = GOOGLdata_ti.iloc[0]['SMA']
    predikt = pred[0]
    if smav > predikt:
        ton = 'Trading stategy: Bear Strategy'
    else:
        ton = 'Trading stategy: Bull Strategy'
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Close Price Prediction: {0:.2f}'.format(predikt), style=style),
        html.Span('SMA: {0:.2f}'.format(smav), style=style),
        html.Span(ton.format(alt), style=style)
    ]

# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    satellite = Orbital('TERRA')
    ts = TimeSeries(key=api_key, output_format='pandas')
    ti = TechIndicators(key=api_key, output_format='pandas')
    period = 100
    GOOGLdata_ts, GOOGLmeta_data_ts = ts.get_intraday(symbol='EBAY', interval='5min', outputsize='compact')
    GOOGLdata_ti, GOOGLmeta_data_ti = ti.get_sma(symbol='EBAY', interval='5min', time_period=period, series_type='close')

    data = {
        'time': [],
        'close price': [],
        'SMA(simple moving average)': [],
        'ML prediction of close price': []
    }

    def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
        """
            series: pd.DataFrame
                dataframe with timeseries

            lag_start: int
                initial step back in time to slice target variable
                example - lag_start = 1 means that the model
                          will see yesterday's values to predict today

            lag_end: int
                final step back in time to slice target variable
                example - lag_end = 4 means that the model
                          will see up to 4 days back in time to predict today

            test_size: float
                size of the test dataset after train/test split as percentage of dataset

            target_encoding: boolean
                if True - add target averages to the dataset

        """

        # copy of the initial dataset
        data = pd.DataFrame(series.copy())
        data.columns = ["y"]

        # lags of series
        for i in range(lag_start, lag_end):
            data["lag_{}".format(i)] = data.y.shift(-i)

        # datetime features
        # train-test split
        y = data.dropna().y
        X = data.dropna().drop(['y'], axis=1)

        return X, y

    X, y = prepareData(GOOGLdata_ts['4. close'], lag_start=0, lag_end=19, test_size=0.3, target_encoding=True)

    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X)

    pred = model.predict(X_test_scaled)

    for j in range(70):
        tme = GOOGLdata_ts.index[j]
        data['time'].append(tme)
        cp = GOOGLdata_ts.iloc[j]['4. close']
        data['close price'].append(cp)
        predplot = pred[j] + 0.1
        data['ML prediction of close price'].append(predplot)
        sma = GOOGLdata_ti.iloc[j]['SMA']
        data['SMA(simple moving average)'].append(sma)

    print(data)
    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    fig.append_trace({
        'x': data['time'],
        'y': data['SMA(simple moving average)'],
        'name': 'SMA',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': data['time'],
        'y': data['ML prediction of close price'],
        'name': 'Prediction',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    '''fig.append_trace({
        'x': data['time'],
        'y': data['Latitude'],
        'text': data['time'],
        'name': 'Longitude vs Latitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 2, 1)'''

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
