import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools
import statsmodels.graphics.tsaplots
from datetime import datetime, timedelta
import time

data = pd.read_csv("totalPrice.csv")
data = data.set_index('Code').T
data.index = pd.to_datetime(data.index)

#Global imports
plt.style.use('seaborn')


#create a series for the 1-lag difference
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    statsmodels.graphics.tsaplots.plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    statsmodels.graphics.tsaplots.plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

def testStationarity(time_series):
    dftest = statsmodels.tsa.stattools.adfuller(time_series)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    return dfoutput


def proper_model(data_ts, maxLag_p = 5, maxLag_q = 5):
    init_aic = sys.maxsize
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag_p):
        for q in np.arange(maxLag_q):
            try:
                model = statsmodels.tsa.arima_model.ARMA(data_ts, order=(p, q), freq = 'D')
                results_ARMA = model.fit(disp=-1, method='css')
                aic = results_ARMA.aic
                if aic < init_aic:
                    init_p = p
                    init_q = q
                    init_properModel = results_ARMA
                    init_aic = aic
            except:
                continue
            
    return init_properModel, init_p, init_q

def diff_to_stationary(ts):
    if(testStationarity(ts)['p-value'] <= 0.05):
        return ts,0
    else:
        ts_diff = ts.diff(1).dropna()
        num = 1
        while(testStationarity(ts_diff)['p-value'] > 0.05):
            ts_diff = ts_diff.diff(1).dropna()
            num += 1
        return ts_diff, num


def arima_predict(ric, start_time, training_days = timedelta(days = 30)):
    start = time.time()
    close_price = data[ric]
    close_price = close_price.reset_index(drop=False)
    close_price.rename(columns={'index':'DATE', ric:'CLOSE'}, inplace = True)

    close_price['DATE'] = pd.to_datetime(close_price['DATE'])

    helper = pd.DataFrame({'DATE': pd.date_range(close_price['DATE'].min(), close_price['DATE'].max())})
    close_price = pd.merge(close_price, helper, on='DATE', how='outer').sort_values('DATE')
    close_price['CLOSE'] = close_price['CLOSE'].interpolate(method='linear')   
    close_price.set_index(pd.to_datetime(close_price.DATE), inplace=True) # set the index to be the DATE
    close_price.sort_index(inplace=True)  # sort the dataframe by the newly created datetime index
    

    
    close_price = close_price[close_price.index >= start_time]
    close_price = close_price[close_price.index <= start_time + training_days]
    ts = close_price['CLOSE']
    ts.index = pd.to_datetime(close_price.index)

    #decomposition = seasonal_decompose(ts, model="additive")
    #trend = decomposition.trend
    #seasonal = decomposition.seasonal
    #residual = decomposition.resid
    #trend.plot()
    #seasonal.plot()
    #residual.plot()
    
    ts_diff, num_of_diff = diff_to_stationary(ts)

    #draw_acf_pacf(ts_diff)

    inf_lst = proper_model(ts_diff)
    
    end_time = start_time + training_days
    day1 = start_time + training_days + timedelta(days = 1)
    day2 = start_time + training_days + timedelta(days = 2)
    day3 = start_time + training_days + timedelta(days = 3)
    day4 = start_time + training_days + timedelta(days = 4)
    day5 = start_time + training_days + timedelta(days = 5)
    
    
    predict_ts = inf_lst[0].predict(day1, day5, dynamic=True)
    
    for i in range(num_of_diff):
        if(num_of_diff - i - 1 != 0):
            predict_ts[day1] = predict_ts[day1] + ts.diff(num_of_diff - i)[end_time]
        else:
            predict_ts[day1] = predict_ts[day1] + ts[end_time]
        predict_ts[day2] = predict_ts[day2] + predict_ts[day1]
        predict_ts[day3] = predict_ts[day3] + predict_ts[day2]
        predict_ts[day4] = predict_ts[day4] + predict_ts[day3]
        predict_ts[day5] = predict_ts[day5] + predict_ts[day4]
    
    
    
    #expected_return = predict_ts['2019-01-02':'2019-01-06']
    #expected_return['2019-01-06'] = (predict_ts['2019-01-06'] - predict_ts['2019-01-05']) /  predict_ts['2019-01-05']
    #expected_return['2019-01-05'] = (predict_ts['2019-01-05'] - predict_ts['2019-01-04']) /  predict_ts['2019-01-04']
    #expected_return['2019-01-04'] = (predict_ts['2019-01-04'] - predict_ts['2019-01-03']) /  predict_ts['2019-01-03']
    #expected_return['2019-01-03'] = (predict_ts['2019-01-03'] - predict_ts['2019-01-02']) /  predict_ts['2019-01-02']
    #expected_return['2019-01-02'] = (predict_ts['2019-01-02'] - ts['2019-01-01']) / ts['2019-01-01']
    print(time.time() - start)
    return (predict_ts[day5] - ts[end_time]) / ts[end_time]

print(arima_predict("AAPL UW Equity",datetime(2018,12,1)))

    













