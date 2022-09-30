import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials


assets = ['MSFT','BAC','F','AA','AXP','BA','CAT','CSCO','CVX','AMZN','DIS','GE','HD','HPQ','INTC','JNJ','JPM','KO','MCD','COST','TGT','WMT','T','VZ','XOM']

yahoo_financials = YahooFinancials(assets)

data = yahoo_financials.get_historical_price_data(start_date='2021-09-27',
                                                  end_date='2021-09-29',
                                                  time_interval='daily')
print(data)