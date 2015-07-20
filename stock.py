import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as met
import pandas.tseries.offsets as tso
import tushare as ts
import datetime as dt

loss = met.make_scorer(met.mean_absolute_error)
def Fexp(x, nu):
    fx = [x[0]]; w = [1]; theta = 1 - 2 ** nu
    for i in range(1, len(x)):
        fx.append(fx[-1] * theta + x[i])
        w.append(w[-1] * theta + 1)
    return(np.array(fx)/np.array(w))
def WaveX(data, gtheta):
    return(pd.DataFrame(np.array(list(map(Fexp, [data.values for i in range(len(gtheta))], gtheta))).T, index = data.index, columns = gtheta))

print("25% -- Loading Data")
yv = "SH000001"
ylist = {"SH000001": "sh", "SZ399005": "zxb", "SZ399006": "cyb"}
data = pd.read_csv(yv + ".csv", index_col = ["Date"], parse_dates = ["Date"])
putin = ts.get_h_data(yv[2:], start = str(dt.date.today() - dt.timedelta(60)), index = True)[::-1]
updata = pd.DataFrame(putin[["close", "volume"]].values.astype(float), columns = ["Close", "Volume"], index = pd.Index(pd.to_datetime(putin.index), name = "Date"))
putin = ts.get_realtime_quotes(ylist[yv])
updata = updata.append(pd.DataFrame(putin[["price", "volume"]].values.astype(float), columns = ["Close", "Volume"], index = pd.Index(pd.to_datetime(putin["date"].values), name = "Date"))).drop_duplicates()
data.update(updata)
data = data.append(updata).drop_duplicates().sort_index()
data.to_csv(yv + ".csv")
dao = np.log(data).replace([np.inf, -np.inf], np.nan).dropna()

print("50% -- Preparing Data")
period = 15; tratio = 0.6; fullth = 0.01
gtheta = np.arange(0, -14, -0.1); lent = int(dao.shape[0] * tratio)
Xcls, Xvol = WaveX(dao["Close"], gtheta), WaveX(dao["Volume"], gtheta)
DX = pd.concat([Xcls.diff(axis = 1).iloc[:,1:]/np.diff(gtheta), Xvol.diff(axis = 1).iloc[:,1:]/np.diff(gtheta)], axis = 1)
dat, dav = DX.iloc[:lent], DX.iloc[lent:]
yt, y = DX.shift(-period).iloc[:lent, int(np.log2(period+1)*10+1)], DX.shift(-period).iloc[lent:, int(np.log2(period+1)*10+1)]

print("75% -- Deep Neural Network")
from keras.layers.core import Dense
from keras.models import Sequential
from keras.regularizers import l2
nhidlay = 1250; 
dnn = Sequential()
dnn.add(Dense(dat.shape[1], nhidlay, init = "uniform", activation = "tanh"))
for i in range(9):
    dnn.add(Dense(nhidlay, nhidlay, init = "uniform", W_regularizer=l2(0.001), activation = "tanh"))
dnn.add(Dense(nhidlay, 1, init = "uniform", activation = "linear"))
dnn.compile(loss = 'mse', optimizer = "sgd")
dnn.fit(dat.values, yt.values, nb_epoch = 30, batch_size = 32, validation_data = (dav.values[:-period], y.values[:-period]))
dnn.save_weights(yv + "_Period_" + str(period) + "_DNN_" + str(nhidlay) + "_10_T" + str(tratio) + ".HDF5")
yhat = dnn.predict(dav.values).flatten()

print("100% -- Make Investment Strategy")
score = (yhat>0)*np.fmin(1, yhat/(0.26*fullth))
cumreturn = np.exp((dao["Close"][dav.index].diff()[1:] * score[:-1]).cumsum() - 0.002*np.abs(np.diff(score)).cumsum() + 0.0001*(1 - score)[:-1].cumsum()); cumreturn.name = "Return"; 
xt = [dav.index[1]] + [dav[str(i)].index[-1] for i in set(dav.index.year)]
yearly = pd.concat([np.exp(dao["Close"][dav.index] - dao["Close"][dat.index[-1]]), cumreturn], axis = 1).loc[xt]
yearly = pd.concat([yearly, np.exp(np.log(yearly).diff()) - 1], axis = 1)
fig = plt.figure(figsize = (12, 4)); 
np.exp(dao["Close"][dav.index] - dao["Close"][dat.index[-1]]).plot(color = "#0000ff", alpha = 0.5); plt.title("Decision in "+ yv+", Period: "+str(period), fontsize = 11); plt.ylabel(yv+" Index")
np.exp((dao["Close"][dav.index] - dao["Close"][dat.index[-1]])[1:]*score[:-1]).plot(color = "#ff0000", linewidth = 0.4); 
cumreturn.plot(color = "#008800", linewidth = 0.5)
for ti in range(2, len(yearly.index)):
    plt.annotate(str(round(100 * yearly.iloc[ti, 2], 1))+"%", xy = (yearly.index[ti]-tso.QuarterEnd(1), yearly.iloc[ti, 0]), xytext = (yearly.index[ti] - tso.QuarterEnd(1), yearly.iloc[ti-1,0]), size = 6, color = '#0000ff', arrowprops=dict(color='#0000ff', alpha=0.5))
    plt.annotate(str(round(100 * yearly.iloc[ti, 3], 1))+"%", xy = (yearly.index[ti], yearly.iloc[ti, 1]), xytext = (yearly.index[ti], yearly.iloc[ti-1,1]), weight = 1000, size = 6, color = '#008800', arrowprops=dict(color='#00cc00', alpha=0.8))
fig.savefig(yv +"_Period_"+str(period)+".png", format = "png")
print(pd.Series([len(yt), len(y), y.index[0], np.abs(np.sign(yhat[:-period]) - np.sign(y[:-period])).mean()/np.abs(np.sign(yt.mean()) - np.sign(y[:-period])).mean(), np.abs(yhat[:-period] - y[:-period]).mean()/np.abs(yt.mean() - y[:-period]).mean(), np.sum((yhat[:-period] - y[:-period])**2)/np.sum((yt.mean() - y[:-period])**2), np.exp(dao["Close"][dav.index].diff().mean() * 250), np.exp(np.log(cumreturn).diff().mean() * 250), np.abs(np.diff(score)).mean() * 250], index = ["Train Length", "Valid Length", "Valid Begin", "L0 Loss", "L1 Loss", "L2 Loss", "Terminal Index (per year)", "Return Ratio (per year)", "Operation Times (per year)"], name = yv))
print(pd.Series(yhat/(0.26*fullth), index = dav.index))
