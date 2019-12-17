#############################################################
#                                                           #
#   Hizkia William Eben                                     #
#   1706042806                                              #
#                                                           #
#   Analysis and Plot Qualcomm and Mediatek Stock Prices    #
#                                                           #
#   References:                                             #
#   # https://medium.com/python-data/setting-up-a-bollinger-#
#     band-with-python-28941e2fa300                         #
#   # https://scipy-cookbook.readthedocs.io/items/          #
#     BrownianMotion.html                                   #
#                                                           #
#                                                           #
#############################################################


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import numpy
from pylab import plot, show, grid, xlabel, ylabel
from math import sqrt
from scipy.stats import norm


priceMediatek = []
dateMediatek = []
priceQualcomm = []
dateQualcomm = []

# Make function for calls to Yahoo Finance
def get_adj_close(ticker, start, end):
    '''
    A function that takes ticker symbols, starting period, ending period
    as arguments and returns with a Pandas DataFrame of the Adjusted Close Prices
    for the tickers from Yahoo Finance
    '''
    start = start
    end = end
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['Adj Close']
    return pd.DataFrame(info)

def getData(file, date, price, start):
    year = int(start)
    with open(str(file)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        temp = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if line_count % 10 == 0:
                    temp = float(temp / 10)
                    price.append(temp)
                    date.append(year)
                    year += float(1/25)
                    temp = 0
                else:
                    temp += float(row[1])
                line_count += 1


def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)
    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)
    return out
                
getData("Mediatek_2002_daily.csv", dateMediatek, priceMediatek, 2002)
getData("Qualcomm_2002_daily.csv", dateQualcomm, priceQualcomm, 2002)

# Get Adjusted Closing Prices for Mediatek and Qualcomm between 2018-2019
qualcomm = get_adj_close('QCOM', '1/1/2018', '14/12/2019')
mediatek = get_adj_close('2454.TW', '1/1/2018', '14/12/2019')

# Calculate 30 Day Moving Average, Std Deviation, Upper Band and Lower Band
for item in (qualcomm, mediatek):
    item['30 Day MA'] = item['Adj Close'].rolling(window=20).mean()
    # set .std(ddof=0) for population std instead of sample
    item['30 Day STD'] = item['Adj Close'].rolling(window=20).std() 
    item['Upper Band'] = item['30 Day MA'] + (item['30 Day STD'] * 2)
    item['Lower Band'] = item['30 Day MA'] - (item['30 Day STD'] * 2)

qualcomm[['Adj Close', '30 Day MA', 'Upper Band', 'Lower Band']].plot(figsize=(20,10))
plt.title('Qualcomm\n2018 until 2019') 
plt.ylabel('Price (USD)')
mediatek[['Adj Close', '30 Day MA', 'Upper Band', 'Lower Band']].plot(figsize=(20,10))
plt.title('Mediatek\n2018 until 2019') 
plt.ylabel('Price (TWD)')
plt.show()

# The Wiener process parameter.
deltaMed = 3
deltaQual = 7
# Total time.
TMed = 5.0
TQual = 10.0
# Number of steps.
NMed = 50
NQual = 100
# Time step size
dtMed = TMed/NMed
dtQual = TQual/NQual
# Number of realizations to generate.
m = 5
# Create an empty array to store the realizations.
predMediatek = numpy.empty((m,NMed+1))
predQualcomm = numpy.empty((m,NQual+1))
# Initial values of x.
predMediatek[:, 0] = priceMediatek[len(priceMediatek)-1]
predQualcomm[:, 0] = priceQualcomm[len(priceQualcomm)-1]

brownian(predMediatek[:,0], NMed, dtMed, deltaMed, out=predMediatek[:,1:])
brownian(predQualcomm[:,0], NQual, dtQual, deltaQual, out=predQualcomm[:,1:])
tMediatek = numpy.linspace(dateMediatek[len(dateMediatek)-1], dateMediatek[len(dateMediatek)-1]+4, NMed+1)
tQualcomm = numpy.linspace(dateQualcomm[len(dateQualcomm)-1], dateQualcomm[len(dateQualcomm)-1]+4, NQual+1)

# plotting the points 
plt.plot(dateMediatek,priceMediatek, color = 'red')
plt.plot(dateQualcomm,priceQualcomm, color = 'blue')
plt.xlabel('Year') 
plt.ylabel('Price (USD)') 
plt.title('Qualcomm vs Mediatek\n2002 until 2019\ndengan Brownian Motion')

for k in range(m):
    plot(tMediatek, predMediatek[k])
    plot(tQualcomm, predQualcomm[k])
xlabel('Year', fontsize=16)
ylabel('Price(USD)', fontsize=16)
grid(True)
show()

