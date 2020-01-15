import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import pandas as pd
import pandas as pd
import csv
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time
from collections import deque

X = deque(maxlen=200)
Y = deque(maxlen=200)

X.append(202001141600)
Y.append(1399.7695)


app = dash.Dash(__name__)
app.layout = html.Div(
    [

        dcc.Graph(id='live-graph-of-predictions', animate=True),
        dcc.Interval(
            id='graph-update',
            interval= 60000,
            n_intervals = 0
        ),
    ]
)

@app.callback(Output('live-graph-of-predictions', 'figure'),
        [Input('graph-update', 'n_intervals')])


def update_graph_scatter(n):
    #time.sleep(65000)
    '''api_key = ''

    ts = TimeSeries(key=api_key, output_format='pandas')
    ti = TechIndicators(key=api_key, output_format='pandas')
    period = 100
    GOOGLdata_ts, GOOGLmeta_data_ts = ts.get_intraday(symbol='GOOGL', interval='1min', outputsize='full')
    #print(GOOGLmeta_data_ts)

    GOOGLdata_ti, GOOGLmeta_data_ti = ti.get_sma(symbol='GOOGL', interval='1min',
                                        time_period=period, series_type='close')

    GOOGLdata_ts.index = pd.to_datetime(GOOGLdata_ts.index)
    GOOGLdata_ts['Index']= GOOGLdata_ts.index
    GOOGLdf1 = GOOGLdata_ti['SMA']
    GOOGLdf2 = GOOGLdata_ts['4. close']
    timecol = GOOGLdata_ts['Index']


    EBAYdata_ts, EBAYmeta_data_ts = ts.get_intraday(symbol='EBAY', interval='1min', outputsize='full')
    #print(EBAYmeta_data_ts)

    EBAYdata_ti, EBAYmeta_data_ti = ti.get_sma(symbol='EBAY', interval='1min',
                                        time_period=period, series_type='close')
    EBAYdf1 = EBAYdata_ti['SMA']
    EBAYdf2 = EBAYdata_ts['4. close']

    X1 = GOOGLdf1.iloc[0]
    X2 = GOOGLdf2.iloc[0]
    X3 = EBAYdf1.iloc[0]
    X4 = EBAYdf2.iloc[0]
    X5 = timecol.iloc[0]
    PX = list(GOOGLdf1)
    PYY = list(timecol)
    '''
    #interval= 1000
    #X = PYY
    #Y = PX
    #global X
    #global Y

    X.append(X[-1]+1)
    Y.append(Y[-1]*random.uniform(0.6,1.4))



    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
            )

    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                yaxis=dict(range=[min(Y),max(Y)]),)}





if __name__ == '__main__':
    app.run_server(debug=True)
