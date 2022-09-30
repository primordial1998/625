import pandas as pd
import os, sys, glob
import numpy as np
import pandas_datareader as pdr
from scipy import stats
import openpyxl
import time
import xlsxwriter

# read Nasdaq and mrkt cap list
df_NAS = pd.read_excel('/Users/primordial/Desktop/Fall 2021/625/NASDAQ.xlsx');
Symbol = df_NAS['Symbol'];
tic_list = list(Symbol);

# get adj close data
start = '2010-12-31';
end = '2020-12-31';
source = 'yahoo';
df_target = pdr.DataReader(tic_list[0:40], source, start, end)['Adj Close'];
time.sleep(3);
print('******Adj close data get');
# df.to_excel('keeper.xlsx', sheet_name='ALLdata');

# check if data has Nan,if yes replace with next one
I = df_target.isnull().sum();
N = 30;
I = I[I.values == 0][0:N];

# selected 30 stocks
df_target = df_target[I.index];

# calcualte equal weight and cap_weight;
# equal weight
eql_weight = np.ones(N) / N;
cap_weight = df_NAS['Market_Cap'];
cap_weight = df_NAS['Market_Cap'][0:30] / sum(df_NAS['Market_Cap'][0:30]);
# calcualte daily return
df_return = df_target.pct_change();

# merge and get our universe
df_return.iloc[0, 0]
M = len(df_return.iloc[:, 0]);
eql_return = [];
cap_return = [];

for i in range(M):
    eql_return.append(np.matmul(df_return.iloc[i, :].values, eql_weight.T))
    cap_return.append(np.matmul(df_return.iloc[i, :].values, cap_weight.T))

IXIC = pdr.DataReader('^IXIC', source, start, end)['Adj Close'];
IXICr = IXIC.pct_change();

df_return['Nasdaq_r'] = IXICr;
df_return['eql_return'] = eql_return;
df_return['cap_return'] = cap_return;
DF_return = df_return.iloc[1:, :];

print(IXICr)

# write table1 (daily), contains Universe,statistics,skew and kurtosis
writer = pd.ExcelWriter('/Users/primordial/Desktop/Fall 2021/625/HW1_table1.xlsx', engine='xlsxwriter');

DF_return.to_excel(writer, sheet_name='Universe');
Df = DF_return.describe();
Df.to_excel(writer, sheet_name='Nas30_statistics');

M = len(DF_return.iloc[:, 0]);
N = len(DF_return.iloc[0, :]);

skew = [];
kurt = [];
for i in range(0, N):
    stats1 = stats.describe(DF_return.iloc[:, i]);
    skew.append(stats1.skewness);
    kurt.append(stats1.kurtosis);

# cal skew and kurt
data = {'skew': skew, 'kurt': kurt};
Df2 = pd.DataFrame(data);
Df2 = Df2.T;
col_names = DF_return.columns;
Df2.columns = list(col_names);
Df2.to_excel(writer, sheet_name='Nas30_skew_and_kurt');
writer.save();
print('******Output table1');

# table 3 4  (cor cov matrix 32x32 Matrix)
df_read = df_return.iloc[1:, :];
X = df_read.iloc[:, 0:];
X_cov = np.cov(X.T);
X_cor = np.corrcoef(X.T);

# write to table3
writer = pd.ExcelWriter('/Users/primordial/Desktop/Fall 2021/625/HW1_table3.xlsx', engine='xlsxwriter');
temp = pd.DataFrame(X_cov, columns=list(col_names), index=list(col_names));
temp.to_excel(writer, sheet_name='Cov');
writer.save();
print('******Output table3 and 4');

# write to table4
writer = pd.ExcelWriter('/Users/primordial/Desktop/Fall 2021/625/HW1_table4.xlsx', engine='xlsxwriter');
temp = pd.DataFrame(X_cor, columns=list(col_names), index=list(col_names));
temp.to_excel(writer, sheet_name='Cor');
writer.save();

# group by month
Df_m = df_read.set_index(df_read.index).groupby(pd.Grouper(freq='M'));
# table 2 (monthly)
Df_m_return = Df_m.sum();
Df_M = Df_m_return.describe();
col = list(Df_M.columns)
Df_m_skew = Df_m_return.skew();
Df_m_kurt = Df_m_return.apply(pd.DataFrame.kurt);

Df_M = Df_M.append(pd.DataFrame({'skew': Df_m_skew, 'kurt': Df_m_kurt}).T)

# write to excel table2
writer = pd.ExcelWriter('/Users/primordial/Desktop/Fall 2021/625/HW1_table2.xlsx', engine='xlsxwriter');
Df_M.to_excel(writer, sheet_name='M_Statistics');

writer.save();
print('******Output table2');
