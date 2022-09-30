import pandas as pd
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import math
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm

print('(2)a')

df = pd.read_excel('/Users/primordial/Desktop/Fall 2021/625/stock_price_reformat.xlsx')
data = pd.read_csv('/Users/primordial/Desktop/Fall 2021/625/all_dividend_date.csv')
loc1 = df.Date[df.Date == '11/1/2018']
print(loc1)

returns_pct = df.iloc[:, 1:].pct_change()
print(returns_pct)
window_1 = returns_pct.iloc[2749 - 272:2749 - 21, :]
print(window_1)


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 0]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data1 = market_model(x, y)

print(data1)

print('The intercept of Stock 1 is', market_model(x, y)[0], ',and the slope of Stock 1 is', market_model(x, y)[1])


loc2 = df.Date[df.Date == '3/24/2015']
print(loc2)


window_2 = returns_pct.iloc[1838 - 272:1838 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 1]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data2 = market_model(x, y)

print('The intercept of Stock 2 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])

loc3 = df.Date[df.Date == '10/31/2017']
print(loc3)


window_3 = returns_pct.iloc[2496 - 272:2496 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 2]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data3 = market_model(x, y)

print('The intercept of Stock 3 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])

loc4 = df.Date[df.Date == '10/30/2017']
print(loc4)


window_4 = returns_pct.iloc[2495 - 272:2495 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 3]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data4 = market_model(x, y)

print('The intercept of Stock 4 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])

loc5 = df.Date[df.Date == '7/22/2016']
print(loc5)

window_5 = returns_pct.iloc[2174 - 272:2174 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 4]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data5 = market_model(x, y)

print('The intercept of Stock 5 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])


loc6 = df.Date[df.Date == '11/28/2017']
print(loc6)


window_6 = returns_pct.iloc[2515 - 272:2515 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 5]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data6 = market_model(x, y)

print('The intercept of Stock 6 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])

loc7 = df.Date[df.Date == '10/30/2017']
print(loc7)

window_7 = returns_pct.iloc[2495 - 272:2495 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 6]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data7 = market_model(x, y)

print('The intercept of Stock 7 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])

loc8 = df.Date[df.Date == '10/11/2017']
print(loc8)

window_8 = returns_pct.iloc[2482 - 272:2482 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 7]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data8 = market_model(x, y)

print('The intercept of Stock 8 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])

loc9 = df.Date[df.Date == '9/13/2017']
print(loc9)

window_9 = returns_pct.iloc[2462 - 272:2462 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 8]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data9 = market_model(x, y)

print('The intercept of Stock 9 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])

loc10 = df.Date[df.Date == '11/14/2017']
print(loc10)

window_10 = returns_pct.iloc[2506 - 272:2506 - 21, :]


def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_


model = LinearRegression()

data_x = window_1.iloc[:, -3]
data_y = window_1.iloc[:, 9]

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y).reshape((-1, 1))
data10 = market_model(x, y)

print('The intercept of Stock 10 is', market_model(x, y)[0], ',and the slope is', market_model(x, y)[1])

print('(2)b')

df1 = returns_pct.iloc[2749 - 20:2749 + 21, :]


e1 = []
for i in range(0, len(df1)):
    error = df1.iloc[i, 0] - data1[0] - data1[1] * df1.iloc[i, -3]
    e1.append(error)
e1 = np.array(e1).reshape(41, 1)
print(np.mean(e1))


df2 = returns_pct.iloc[1838 - 20:1838 + 21, :]
e2 = []

for i in range(0, len(df2)):
    error = df2.iloc[i, 1] - data2[0] - data2[1] * df2.iloc[i, -3]
    e2.append(error)
e2 = np.array(e2).reshape(41, 1)

df3 = returns_pct.iloc[2496 - 20:2496 + 21, :]
e3 = []

for i in range(0, len(df3)):
    error = df3.iloc[i, 2] - data3[0] - data3[1] * df3.iloc[i, -3]
    e3.append(error)
e3 = np.array(e3).reshape(41, 1)

df4 = returns_pct.iloc[2495 - 20:2495 + 21, :]
e4 = []

for i in range(0, len(df4)):
    error = df4.iloc[i, 3] - data4[0] - data3[1] * df4.iloc[i, -3]
    e4.append(error)
e4 = np.array(e4).reshape(41, 1)

df5 = returns_pct.iloc[2174 - 20:2174 + 21, :]
e5 = []

for i in range(0, len(df5)):
    error = df5.iloc[i, 4] - data5[0] - data5[1] * df5.iloc[i, -3]
    e5.append(error)
e5 = np.array(e5).reshape(41, 1)

df6 = returns_pct.iloc[2515 - 20:2515 + 21, :]
e6 = []

for i in range(0, len(df6)):
    error = df6.iloc[i, 5] - data6[0] - data6[1] * df6.iloc[i, -3]
    e6.append(error)
e6 = np.array(e6).reshape(41, 1)

df7 = returns_pct.iloc[2495 - 20:2495 + 21, :]
e7 = []

for i in range(0, len(df7)):
    error = df7.iloc[i, 6] - data7[0] - data3[1] * df7.iloc[i, -3]
    e7.append(error)
e7 = np.array(e7).reshape(41, 1)

df8 = returns_pct.iloc[2482 - 20:2482 + 21, :]
e8 = []

for i in range(0, len(df8)):
    error = df8.iloc[i, 7] - data8[0] - data8[1] * df8.iloc[i, -3]
    e8.append(error)
e8 = np.array(e8).reshape(41, 1)

df9 = returns_pct.iloc[2462 - 20:2462 + 21, :]
e9 = []

for i in range(0, len(df9)):
    error = df9.iloc[i, 8] - data9[0] - data9[1] * df9.iloc[i, -3]
    e9.append(error)
e9 = np.array(e9).reshape(41, 1)

df10 = returns_pct.iloc[2506 - 20:2506 + 21, :]
e10 = []

for i in range(0, len(df10)):
    error = df10.iloc[i, 9] - data10[0] - data10[1] * df10.iloc[i, -3]
    e10.append(error)
e10 = np.array(e10).reshape(41, 1)

all_e = np.array([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]).reshape(41, 10)
e_bars = all_e.mean(axis=1)
e_total_vars = all_e.var(axis=1)


stock1_CAR = []
for i in range(1, len(df1) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e1[0:i])
    stock1_CAR.append(CAR1)


stock2_CAR = []
for i in range(1, len(df2) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e2[0:i])
    stock2_CAR.append(CAR1)


stock3_CAR = []
for i in range(1, len(df3) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e3[0:i])
    stock3_CAR.append(CAR1)


stock4_CAR = []
for i in range(1, len(df4) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e4[0:i])
    stock4_CAR.append(CAR1)


stock5_CAR = []
for i in range(1, len(df5) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e5[0:i])
    stock5_CAR.append(CAR1)

stock6_CAR = []
for i in range(1, len(df6) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e6[0:i])
    stock6_CAR.append(CAR1)


stock7_CAR = []
for i in range(1, len(df7) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e7[0:i])
    stock7_CAR.append(CAR1)


stock8_CAR = []
for i in range(1, len(df8) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e8[0:i])
    stock8_CAR.append(CAR1)


stock9_CAR = []
for i in range(1, len(df9) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e9[0:i])
    stock9_CAR.append(CAR1)


stock10_CAR = []
for i in range(1, len(df10) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e10[0:i])
    stock10_CAR.append(CAR1)


std1 = e1[19:21].std()
std2 = e2[19:21].std()
std3 = e3[19:21].std()
std4 = e4[19:21].std()
std5 = e5[19:21].std()
std6 = e6[19:21].std()
std7 = e7[19:21].std()
std8 = e8[19:21].std()
std9 = e9[19:21].std()
std10 = e10[19:21].std()


t = []
CAR = []

for i in range(-20, 21, 1):
    t.append(i)
list1 = [t, e1]

for i in range(1, len(df1) + 1):
    n = np.ones(i)
    CAR1 = np.dot(n, e_bars[0:i])
    CAR.append(CAR1)


list1 = [t, e_bars, CAR]
print(pd.DataFrame(list1, index=['Days', 'e_bar', 'CAR']).T)


plt.figure(figsize=(18, 6))

plt.plot(t, e_bars)
plt.xlabel('Days')
plt.ylabel('e_bars')
plt.plot([0, 0], [-0.02, 0.01], '--', color='red')
plt.title('e_bars')
plt.show()


plt.figure(figsize=(18, 6))

plt.plot(t, CAR)
plt.xlabel('Days')
plt.ylabel('CAR')
plt.plot([0, 0], [-0.03, 0.01], '--', color='red')
plt.title('CAR')
plt.show()

print('(2)c')

p = 0.05
from scipy.stats import norm

print('The p-value of 0.95 significant is: ,', norm.ppf(p))

def J(CAR_bar, var_bar):
    return CAR_bar / var_bar


CAR_bar = np.mean(CAR)
var_bar = math.sqrt((1 / 41 ** 2) * np.sum(e_total_vars))

print('The J statistics is:\n',J(CAR_bar, var_bar))

print('Since the J statsitic in the time interval is -9.90859, we therefore reject the null hypothesis.')


print('(2)d')

df1 = returns_pct.iloc[2749 + 1:2749 + 4, :]
e1 = []
for i in range(0, len(df1)):
    error = df1.iloc[i, 0] - data1[0] - data1[1] * df1.iloc[i, -3]
    e1.append(error)
e1 = np.array(e1).reshape(3, 1)

df2 = returns_pct.iloc[1838 + 1:1838 + 4, :]
e2 = []

for i in range(0, len(df2)):
    error = df2.iloc[i, 1] - data2[0] - data2[1] * df2.iloc[i, -3]
    e2.append(error)
e2 = np.array(e2).reshape(3, 1)


df3 = returns_pct.iloc[2496 + 1:2496 + 4, :]
e3 = []

for i in range(0, len(df3)):
    error = df3.iloc[i, 2] - data3[0] - data3[1] * df3.iloc[i, -3]
    e3.append(error)
e3 = np.array(e3).reshape(3, 1)

df4 = returns_pct.iloc[2495 + 1:2495 + 4, :]
e4 = []

for i in range(0, len(df4)):
    error = df4.iloc[i, 3] - data4[0] - data3[1] * df4.iloc[i, -3]
    e4.append(error)
e4 = np.array(e4).reshape(3, 1)

df5 = returns_pct.iloc[2174 + 1:2174 + 4, :]
e5 = []

for i in range(0, len(df5)):
    error = df5.iloc[i, 4] - data5[0] - data5[1] * df5.iloc[i, -3]
    e5.append(error)
e5 = np.array(e5).reshape(3, 1)

df6 = returns_pct.iloc[2515 + 1:2515 + 4, :]
e6 = []

for i in range(0, len(df6)):
    error = df6.iloc[i, 5] - data6[0] - data6[1] * df6.iloc[i, -3]
    e6.append(error)
e6 = np.array(e6).reshape(3, 1)

df7 = returns_pct.iloc[2495 + 1:2495 + 4, :]
e7 = []

for i in range(0, len(df7)):
    error = df7.iloc[i, 6] - data7[0] - data3[1] * df7.iloc[i, -3]
    e7.append(error)
e7 = np.array(e7).reshape(3, 1)

df8 = returns_pct.iloc[2482 + 1:2482 + 4, :]
e8 = []

for i in range(0, len(df8)):
    error = df8.iloc[i, 7] - data8[0] - data8[1] * df8.iloc[i, -3]
    e8.append(error)
e8 = np.array(e8).reshape(3, 1)

df9 = returns_pct.iloc[2462 + 1:2462 + 4, :]
e9 = []

for i in range(0, len(df9)):
    error = df9.iloc[i, 8] - data9[0] - data9[1] * df9.iloc[i, -3]
    e9.append(error)
e9 = np.array(e9).reshape(3, 1)

df10 = returns_pct.iloc[2506 + 1:2506 + 4, :]
e10 = []

for i in range(0, len(df10)):
    error = df10.iloc[i, 9] - data10[0] - data10[1] * df10.iloc[i, -3]
    e10.append(error)
e10 = np.array(e10).reshape(3, 1)

all_e = np.array([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]).reshape(3, 10)
e_bars = all_e.mean(axis=1)
e_vars = all_e.var(axis=1)

CAR_bar = np.mean(CAR[21:24])
var_bar = math.sqrt((1 / 3 ** 2) * np.sum(e_vars))
print('The J statistics is :\n',J(CAR_bar, var_bar))

print('The J statistics is -2.51378 which is statistically significant.')


print('(2)e')

CAR_bar = stock1_CAR[20]
var_bar = std1
print('The J statistic for stock1 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock2_CAR[20]
var_bar = std2
print('The J statistic for stock2 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock3_CAR[20]
var_bar = std3
print('The J statistic for stock3 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock4_CAR[20]
var_bar = std4
print('The J statistic for stock4 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock5_CAR[20]
var_bar = std5
print('The J statistic for stock5 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock6_CAR[20]
var_bar = std6
print('The J statistic for stock6 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock7_CAR[20]
var_bar = std7
print('The J statistic for stock7 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock8_CAR[20]
var_bar = std8
print('The J statistic for stock8 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock9_CAR[20]
var_bar = std9
print('The J statistic for stock9 at the announcement date is:', J(CAR_bar, var_bar))

CAR_bar = stock10_CAR[20]
var_bar = std10
print('The J statistic for stock10 at the announcement date is:', J(CAR_bar, var_bar))

print('Since the threshold is abs(1.6448536), it is clear that except stock1, stock 7, stock9, other stocks are statistically significant.')

