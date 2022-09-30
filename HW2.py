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

start = '2010-12-31'
end = '2021-01-01'
yf.pdr_override()
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'ABT', 'V', 'JPM', 'JNJ', 'WMT',
           'UNH', 'PG', 'HD', 'BAC', 'TMO', 'MA', 'DIS', 'ADBE', 'CMCSA', 'KO', 'CRM', 'NFLX',
           'NKE', 'ORCL', 'CSCO', 'LLY', 'DHR', 'XOM', 'VZ', '^IXIC']
prices_df = yf.download(tickers, start, end, progress=True, auto_adjust=True)['Close']

import numpy as np

returns_pct = prices_df.pct_change()  # daily percentage return
print('The total returns of individual stocks is:\n', returns_pct.sum())

print(returns_pct)

df = pd.read_excel('/Users/primordial/Desktop/Fall 2021/625/NASDAQ.xlsx')
print(df)

daily_return_matrix_pcw = np.dot(returns_pct.iloc[:, 0:30], df['Weight'][0:30])
cap_weight = df['Weight'][0:30]
wdr = pd.DataFrame(daily_return_matrix_pcw)
print('Total market cap weighted portfolio daily return is:', daily_return_matrix_pcw[1:2518].sum())
print(wdr)

daily_return_matrix_eqw = np.dot(returns_pct.iloc[:, 0:30], 1 / 30 * np.ones(30))
eql_weight = 1 / 30 * np.ones(30)
ewdr = pd.DataFrame(daily_return_matrix_eqw)
print('Total equally weighted portfolio daily return is:', daily_return_matrix_eqw[1:2518].sum())
print(ewdr)

returns_matrix = pd.DataFrame(returns_pct)
returns_matrix['Cap Weighted Port'] = wdr.values
returns_matrix['Equal Weighted Port'] = ewdr.values

print(returns_matrix)

monthly_returns_matrix = returns_matrix.resample('M').agg(lambda x: (x + 1).prod() - 1)
monthly_returns_stats = monthly_returns_matrix.agg(['count', 'mean', 'std', 'skew', 'kurt']).T
print(monthly_returns_stats)

cov_matrix = returns_matrix.cov()
print(cov_matrix)

print('the correlation matrix is: \n', returns_matrix.corr())


# 1a

def cov_beta(cov_self_market, var_market):
    return cov_self_market / var_market


data = []

for i in range(len(cov_matrix)):
    cov_self_market = cov_matrix.iloc[i, 30]
    cov_self_market
    sigma_market = cov_matrix.iloc[30, 30]
    data.append(cov_beta(cov_self_market, sigma_market))

index = ['AAPL', 'ABT', 'ADBE', 'AMZN', 'BAC', 'CMCSA', 'CRM', 'CSCO', 'DHR',
         'DIS', 'GOOGL', 'HD', 'JNJ', 'JPM', 'KO', 'LLY', 'MA', 'MSFT', 'NFLX',
         'NKE', 'NVDA', 'ORCL', 'PG', 'TMO', 'TSLA', 'UNH', 'V', 'VZ', 'WMT',
         'XOM', '^IXIC', 'Cap Weighted Port', 'Equal Weighted Port']
data = pd.DataFrame(data, index=index)
data.columns = ['beta']
print(data)

# 1b beta for single asset is equal the coeeficient of linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()

def market_model(x, y):
    model.fit(x, y)
    return model.intercept_, model.coef_

for i in range(len(returns_matrix.T)):
    data_x = returns_matrix.iloc[1:, i]  # Stock Return
    data_y = returns_matrix.iloc[1:, 30]  # Market Return
    x = np.array(data_x).reshape((-1, 1))
    y = np.array(data_y).reshape((-1, 1))
    print('The Alpha and Beta for stocks or portfolio is\n', market_model(x, y))

#2a,2b
print(monthly_returns_matrix)

monthly_cov_matrix = monthly_returns_matrix.cov()
print('monthly covariance matrix is :\n',monthly_cov_matrix)
var_ewp = monthly_cov_matrix.iloc[32,32]
var_cwp = monthly_cov_matrix.iloc[31,31]
print('variance of equally weighted and cap weighted portfolios are',var_ewp,'and',var_cwp,'respectively')

beta_ewp = monthly_cov_matrix.iloc[30,32]/monthly_cov_matrix.iloc[30,30]
beta_cwp = monthly_cov_matrix.iloc[31,32]/monthly_cov_matrix.iloc[30,30]
print('beta of equally weighted and cap weighted portfolios are',beta_ewp,'and',beta_cwp,'respectively')

#2c
#beta of equally weighted portfolio is higher than cap weighted portfofio, therefore it has higher systematic risk, but the variance of
# equally weighted portfolio is lower than cap-weighted portfolio, therefore it has lower total risk.

#3 cooperate with Li Yu

data_e = returns_pct.iloc[:,:-3]

mus = (1+data_e.mean())**252 - 1

cov = data_e.cov()*252

print(mus,cov)

# - How many assests to include in each portfolio
n_assets = 30
# -- How many portfolios to generate
n_portfolios = 1000

# -- Initialize empty list to store mean-variance pairs for plotting
mean_variance_pairs = []

np.random.seed(0)
# -- Loop through and generate lots of random portfolios
for i in range(n_portfolios):
    # - Choose assets randomly without replacement
    assets = np.random.choice(list(data_e.columns), n_assets, replace=False)
    # - Choose weights randomly
    weights = np.random.rand(n_assets)
    # - Ensure weights sum to 1
    weights = weights / sum(weights)

    # -- Loop over asset pairs and compute portfolio return and variance
    portfolio_E_Variance = 0
    portfolio_E_Return = 0
    for i in range(len(assets)):
        portfolio_E_Return += weights[i] * mus.loc[assets[i]]
        for j in range(len(assets)):
            # -- Add variance/covariance for each asset pair
            # - Note that when i==j this adds the variance
            portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]

    # -- Add the mean/variance pairs to a list for plotting
    mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])

#-- Plot the risk vs. return of randomly generated portfolios
#-- Convert the list from before into an array for easy plotting
mean_variance_pairs = np.array(mean_variance_pairs)

risk_free_rate=0 #-- Include risk free rate here

fig = go.Figure()
fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0],
                      marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5),
                                  showscale=True,
                                  size=7,
                                  line=dict(width=1),
                                  colorscale="RdBu",
                                  colorbar=dict(title="Sharpe<br>Ratio")
                                 ),
                      mode='markers'))
fig.update_layout(template='plotly_white',
                  xaxis=dict(title='Annualised Risk (Volatility)'),
                  yaxis=dict(title='Annualised Return'),
                  title='Sample of Random Portfolios',
                  width=850,
                  height=500)
fig.update_xaxes(range=[0.16, 0.22])
fig.update_yaxes(range=[0.15,0.27])
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))

#-- Create random portfolio weights and indexes
#- How many assests in the portfolio
n_assets = 30

mean_variance_pairs = []
weights_list=[]
tickers_list=[]

for i in tqdm(range(10000)):
    next_i = False
    while True:
        #- Choose assets randomly without replacement
        assets = np.random.choice(list(data_e.columns), n_assets, replace=False)
        #- Choose weights randomly ensuring they sum to one
        weights = np.random.rand(n_assets)
        weights = weights/sum(weights)

        #-- Loop over asset pairs and compute portfolio return and variance
        portfolio_E_Variance = 0
        portfolio_E_Return = 0
        for i in range(len(assets)):
            portfolio_E_Return += weights[i] * mus.loc[assets[i]]
            for j in range(len(assets)):
                portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]

        #-- Skip over dominated portfolios
        for R,V in mean_variance_pairs:
            if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                next_i = True
                break
        if next_i:
            break

        #-- Add the mean/variance pairs to a list for plotting
        mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])
        weights_list.append(weights)
        tickers_list.append(assets)
        break

#-- Plot the risk vs. return of randomly generated portfolios
#-- Convert the list from before into an array for easy plotting
mean_variance_pairs = np.array(mean_variance_pairs)

risk_free_rate=0 #-- Include risk free rate here

fig = go.Figure()
fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0],
                      marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5),
                                  showscale=True,
                                  size=7,
                                  line=dict(width=1),
                                  colorscale="RdBu",
                                  colorbar=dict(title="Sharpe<br>Ratio")
                                 ),
                      mode='markers',
                      text=[str(np.array(tickers_list[i])) + "<br>" + str(np.array(weights_list[i]).round(2)) for i in range(len(tickers_list))]))
fig.update_layout(template='plotly_white',
                  xaxis=dict(title='Annualised Risk (Volatility)'),
                  yaxis=dict(title='Annualised Return'),
                  title='Sample of Random Portfolios',
                  width=850,
                  height=500)
fig.update_xaxes(range=[0.16, 0.22])
fig.update_yaxes(range=[0.15,0.27])
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))

returns_matrix.sum()   #total returns over 10 years
returns_monthly = data_e.sum()/(10*12)  #returns anually
cov_monthly = data_e.cov()/((10*12)**2)
print(returns_monthly.shape,  cov_monthly.shape)
#annualy covariance matrix
def min_var_given_return(A,B,C,e):
    upper_minvar = A*e**2-2*B*e+C
    lower_minvar = A*C-B**2
    return upper_minvar/lower_minvar

cov_inv = np.linalg.pinv(cov.values)
upper_gm = cov_inv.dot(np.ones(len(cov)))
A = lower_gm = ((np.ones(len(cov)).T).dot(cov_inv)).dot(np.ones(len(cov)))
global_minvar_weight = upper_gm/lower_gm
gm_returns = data_e.dot(global_minvar_weight)
gm_returns.agg(['mean','std'])

A = ((np.ones(len(cov)).T).dot(cov_inv)).dot(np.ones(len(cov)))
B = ((np.ones(len(cov)).T).dot(cov_inv)).dot(returns_monthly)
C = (returns_monthly.dot(cov_inv)).dot(returns_monthly.T)
mu = returns_monthly


def z(A, B, C, cov_inv, mu):
    upper_z = C * cov_inv.dot(np.ones(len(cov))) - B * cov_inv.dot(mu)
    lower_z = A * C - B ** 2
    return upper_z / lower_z


def h(A, B, C, cov_inv, mu):
    upper_h = A * cov_inv.dot(mu) - B * cov_inv.dot(np.ones(len(cov)))
    lower_h = A * C - B ** 2
    return upper_h / lower_h


e1 = np.linspace(0, 0.5, 100)
std1 = []
w_p = []
std_p = monthly_returns_stats['std'].iloc[:-3].values

for e in e1:
    w_p = (z(A, B, C, cov_inv, mu) + h(A, B, C, cov_inv, mu) * e)  # Compute the optimal weight
    std1.append(np.dot(w_p, std_p))

    # std1.append(np.array(w_p).dot(np.array(std_p)))

plt.figure(figsize=(16, 10))
plt.plot(std1, e1, label='Efficient Frontier')
plt.xlabel('Monthly Standard Deviation', fontsize=20)
plt.ylabel('Monthly return', fontsize=20)
plt.legend()

#3b
def min_var_given_return(A,B,C,e):
    upper_minvar = A*e**2-2*B*e+C
    lower_minvar = A*C-B**2
    return upper_minvar/lower_minvar

e = 0.05 ##Monthly return
A = ((np.ones(len(cov)).T).dot(cov_inv)).dot(np.ones(len(cov)))
B = ((np.ones(len(cov)).T).dot(cov_inv)).dot(returns_monthly)
C = (returns_monthly.dot(cov_inv)).dot(returns_monthly.T)
D = A*C-B**2


#print(min_var_given_return(A,B,C,e))

e = 0.05
w_p = z(A,B,C,cov_inv,mu)+h(A,B,C,cov_inv,mu)*e
std_p = w_p.dot(monthly_returns_stats['std'].iloc[:-3])
print('The standard deviation of a efficient portfolio with monthly return of 0.05 is:',std_p)

#3c
slope = (0.5-0.000417)/(std1[-1]-0.008315)
return_q = (0.25-0.008315)*slope+0.000417
print('The portfolio mean return given std = 0.25 is:',return_q)

#3d
cov_pq = (A*(e-B/A)*(return_q-B/A))/D+1/A
print('The covariance between portfolio P and portfolio Q is:',cov_pq)

#3e
r_f = 0.00005
tan_return = (C-r_f*B)/(B-r_f*A)
var_tan = (C-2*r_f*B+r_f**2*B)/((B-r_f*A)**2)
std_tan = math.sqrt(var_tan)
tan_slope = (tan_return-r_f)/std_tan

print('The tangency portfolio return is:',tan_return,'\n The tangency portfolio std is:',std_tan,'\n The slope is:',tan_slope)
