import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

import numpy as np                               # vectors and matrices
import pandas as pd
from pandas import read_csv                            # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
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
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
scaler = StandardScaler()
from sklearn.externals import joblib

#%matplotlib inline

data1 = pd.read_csv('finaltweetings.csv')
data2 = pd.read_csv('stocks.csv')

merged = data1.combine_first(data2)
merged.index = pd.to_datetime(merged["date"])
f1 = plt.figure()
merged.plot(subplots=True, figsize=(15,6))
plt.draw()  # Draws, but does not block

f2 = plt.figure()
sns.heatmap(merged.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
plt.gcf()
plt.title("correlation plot")
plt.draw()

#multiplicative decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_csv
#series = [i+randrange(10) for i in range(1,100)]
#f3 = plt.figure()
#result = seasonal_decompose(merged, model='additive', freq=1)
#result.plot()
series1 = merged[['sentscor','GOOGLClose']] #read_csv('airline-passengers.csv', header=0, index_col=0)
series1.index = pd.to_datetime(merged["date"])
series1 = series1.dropna()
#print(series1)
result = seasonal_decompose(series1, model='additive',freq=1)
result.plot()


Stock = pd.read_csv('stocks.csv', index_col=['date'], parse_dates=['date'])
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
    data.index = pd.to_datetime(data.index)
    # train-test split
    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)

    def timeseries_train_test_split(X, y, test_size):
        """
            Perform train-test split with respect to time series structure
        """

        # get the index after which test set starts
        test_index = int(len(X)*(1-test_size))

        X_train = X.iloc[:test_index]
        y_train = y.iloc[:test_index]
        X_test = X.iloc[test_index:]
        y_test = y.iloc[test_index:]

        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepareData(Stock.EBAYClose, lag_start=6, lag_end=25, test_size=0.3, target_encoding=True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)


plt.figure(figsize=(10, 8))
plt.title('correlation heatmap')
sns.heatmap(X_train.corr());

tscv = TimeSeriesSplit(n_splits=3)
lasso = LassoCV(cv=tscv)
lasso.fit(X_train_scaled, y_train)

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled, y_train)


joblib_file = "ML_Model3.pkl"
#joblib.dump(lr, joblib_file)

def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies

    """

    prediction = model.predict(X_test)

    plt.figure(figsize=(15, 7))

    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    plt.draw()
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                                    cv=tscv,
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        plt.draw()

        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
            plt.draw()
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.draw()

plotModelResults(lasso,
                 X_train=X_train_scaled,
                 X_test=X_test_scaled,
                 plot_intervals=True, plot_anomalies=True)


def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(15, 7))
    plt.title('sorted coefficient values of the model')
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    plt.draw()
plotCoefficients(lasso)

plt.show()
