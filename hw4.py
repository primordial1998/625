import pandas as pd
import os, sys, glob
import numpy as np
import pandas_datareader as pdr
from scipy import stats
import openpyxl
import time
import xlsxwriter
import yfinance as yf
from yahoofinancials import YahooFinancials

rf = pd.read_excel('/Users/primordial/Desktop/Fall 2021/625/daily_int_rate.xlsx')
rf = rf.set_index('DATE').resample('M').asfreq()
print(rf)

start = '2007-12-03'
end = '2021-09-30'

tickers = ['AAPL','GOOG','IBM','MSFT','BAC','F','GM','AA','AXP','BA','CAT',
          'CSCO','CVX','AMZN','DIS','GE','HD','HPQ','INTC','JNJ','JPM','KO','MCD',
           'COST','TGT','WMT','T','VZ','PSX','XOM','^GSPC']
stockPrice = yf.download(tickers, start, end, progress=True, auto_adjust=True)['Close']
stock_pct = stockPrice.pct_change().resample('M').agg(lambda x: (x + 1).prod() - 1)

print(stock_pct)
