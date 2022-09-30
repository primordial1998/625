import numpy as np
import pandas as pd
import seaborn as sns
import math
import os
import sys
import random
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import openpyxl
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import scipy as sp


def tree_to_table(tree, name_str):
    n_gen = len(tree)
    n_chrom = len(tree[0])
    name = []
    gen = []
    chrom = []
    ret = []
    vol = []
    sr = []
    res = []
    for i in range(n_gen):
        for j in range(n_chrom):
            name.append(name_str)
            gen.append(i)
            chrom.append(j)
            ret.append(tree[i][j].get_expected_return())
            vol.append(tree[i][j].get_volatility())
            sr.append(tree[i][j].get_sharpe_ratio())
    d = {'name': name, 'generation': gen, 'chromosome': chrom, 'return': ret, 'volatility': vol, 'sharpe_ratio': sr}
    df = pd.DataFrame(data=d)
    return df

class Portfolio:
    def __init__(self, returns_df, corr_matrix, risk_free):
        self.returns = returns_df
        self.corr = corr_matrix
        self.description = self.returns.describe()
        self.n_assets = self.returns.shape[1]
        self.risk_free = risk_free
        self.initialize_weights()
        self.set_expected_return()
        self.set_volatility()
        self.set_sharpe_ratio()
        return None
    def initialize_weights(self):
        self.weights = np.random.rand(self.n_assets)
        self.weights = (self.weights/self.weights.sum())
        return None
    def set_weights(self, weights):
        self.weights = (weights/weights.sum())
        self.set_discrete_weights = self.weights
        return None
    def set_expected_return(self):
        weighted = self.weights*self.description.loc['mean']
        self.expected_return = weighted.sum().round(2)
        return None
    def set_volatility(self):
        std = self.description.loc['std'].values
        m1 = (self.weights*std).reshape(self.n_assets,1)
        m2 = m1.reshape(1,self.n_assets)
        self.volatility = math.sqrt((m1*self.corr*m2).sum())
        return None
    def set_sharpe_ratio(self):
        self.sharpe_ratio = (self.expected_return-self.risk_free)/self.volatility
        return None
    def get_weights(self):
        return self.weights
    def get_expected_return(self):
        return self.expected_return
    def get_volatility(self):
        return self.volatility
    def get_sharpe_ratio(self):
        return self.sharpe_ratio

class Genetic_algorithm:
    def __init__(self, returns_df, corr_matrix, risk_free, size):
        self.returns = returns_df
        self.corr = corr_matrix
        self.risk_free = risk_free
        self.size = size
        return None
    def run(self, iterations, variable):
        self.initialize()
        for i in range(iterations):
            self.set_fitness(variable)
            self.select_fittest()
            self.crossover()
            self.mutation()
            self.pass_generation(variable)
        return None
    def initialize(self):
        self.population = []
        self.offspring = []
        self.population_best = []
        self.population_fitness = []
        self.population_mean  = []
        self.tree = list()
        for i in range(self.size):
            self.population.append(Portfolio(self.returns, self.corr, self.risk_free))
        self.population_df = self.to_table(self.population)
        self.tree.append(self.population)
        return None
    def set_fitness(self, variable):
        if(variable == 'volatility'):
            max_volatility = self.population_df[variable].max()
            self.population_df.sort_values(by=variable, inplace=True, ascending=True)
            self.population_df['fitness'] = max_volatility - self.population_df[variable] + 1
            self.population_df['fitness'] = self.population_df['fitness']/self.population_df['fitness'].sum()
        else:
            self.population_df.sort_values(by=variable, inplace=True, ascending=False)
            self.population_df['fitness'] = self.population_df[variable]/self.population_df[variable].sum()
        self.population_df['selection_prob'] = self.population_df['fitness']
        for i in range(1, len(self.population_df['selection_prob'])):
            self.population_df['selection_prob'].iloc[i] = self.population_df['selection_prob'].iloc[i-1] + self.population_df['selection_prob'].iloc[i]
        return self.population_df
    def select_fittest(self, rand = True):
        third = int(self.size/3)
        idx = self.population_df.head(third).index.values
        if(rand == True):
            for i in idx:
                self.offspring.append(self.population[i])
                p = Portfolio(self.returns, self.corr, self.risk_free)
                self.offspring.append(p)
        else:
            for i in idx:
                self.offspring.append(self.population[i])
        return None
    def crossover(self):
        rest = self.size - len(self.offspring)
        for i in range(rest):
            idx_parent1 = self.select_parent()
            idx_parent2 = self.select_parent()
            alpha = random.random()
            w3 = alpha*self.population[idx_parent1].get_weights() + (1-alpha)*self.population[idx_parent2].get_weights()
            p = Portfolio(self.returns, self.corr, self.risk_free)
            p.set_weights(w3)
            self.offspring.append(p)
        return None
    def mutation(self):
        n_assets = len(self.population[0].get_weights())
        for child in self.offspring:
            idx1 = random.randrange(0,n_assets)
            idx2 = random.randrange(0,n_assets)
            w = child.get_weights()
            minimo = min(w[idx1],w[idx2])
            rand = random.uniform(0,minimo)
            w[idx1] += rand
            w[idx2] -= rand
            child.set_weights(w)
        return None
    def pass_generation(self, variable):
        self.population = self.offspring
        self.offspring = []
        self.population_df = self.to_table(self.population)
        self.tree.append(self.population)
        best = self.population[0]
        mean_fit = self.population_df[variable].mean()
        if(variable == 'volatility'):
            max_fit = self.population_df.sort_values(by=variable, ascending=True).head(1)[variable].iloc[0]
        else:
            max_fit = self.population_df.sort_values(by=variable, ascending=False).head(1)[variable].iloc[0]
        self.population_best.append(best)
        self.population_fitness.append(max_fit)
        self.population_mean.append(mean_fit)
        return None
    def select_parent(self):
        roulette = random.random()
        i = 0
        while roulette > self.population_df['selection_prob'].iloc[i]:
            i += 1
        if(i > 0):
            i = i - 1
        return self.population_df.iloc[i,:].name
    def to_table(self, array):
        exp_returns = [s.get_expected_return() for s in array]
        volatilities = [s.get_volatility() for s in array]
        sharpe_ratios = [s.get_sharpe_ratio() for s in array]
        d = {'return': exp_returns, 'volatility': volatilities, 'sharpe_ratio' : sharpe_ratios}
        df = pd.DataFrame(data=d)
        return df
    def get_samples(self):
        return self.samples
    def get_population(self,variable):
        self.population_df.sort_values(by=variable, inplace=True, ascending=False)
        idx = self.population_df.head(1).index.values[0]
        return self.population[idx]
    def get_tree(self, variable):
        return self.tree
    def get_analysis(self):
        d = {'best_fitness':self.population_fitness,'fitness_mean':self.population_mean}
        output = pd.DataFrame(data=d)
        return output



start = '2012-10-31'
end = '2021-10-31'
yf.pdr_override()
# tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'IBM', 'CAT', 'T', 'V', 'JPM', 'JNJ', 'WMT',
#            'F', 'INTC', 'HD', 'BAC', 'TGT', 'KO', 'DIS', 'BA', 'AA', 'AXP', 'CVX', 'GM',
#            'HPQ', 'GE', 'CSCO','COST', 'XOM', 'VZ', 'MCD']

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'IBM', 'CAT', 'T', 'V', 'JPM', 'JNJ', 'WMT',
            'F', 'INTC', 'HD', 'BAC', 'TGT', 'KO', 'DIS', 'BA', 'AA', 'AXP', 'CVX', 'GM',
            'HPQ', 'GE', 'CSCO','COST', 'XOM', 'VZ', 'MCD']

benchmark_ticker = ['^GSPC']
stock_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'IBM', 'CAT', 'T', 'V', 'JPM', 'JNJ', 'WMT',
            'F', 'INTC', 'HD', 'BAC', 'TGT', 'KO', 'DIS', 'BA', 'AA', 'AXP', 'CVX', 'GM',
            'HPQ', 'GE', 'CSCO','COST', 'XOM', 'VZ', 'MCD']

prices_df = yf.download(tickers, start, end, progress=True, auto_adjust=True)['Close']
returns_pct = prices_df.pct_change()

risk_free = 0.0143
returns_matrix = pd.DataFrame(returns_pct)
monthly_returns = returns_matrix.resample('M').agg(lambda x: (x + 1).prod() - 1)
monthly_returns_stats = monthly_returns.agg(['count', 'mean', 'std', 'skew', 'kurt']).T
print(monthly_returns_stats)

corr = monthly_returns.corr()
corr_matrix = corr.values

##

sns.set()
plt.figure(figsize=(15,8))
mask = np.triu(np.ones_like(corr, dtype=bool))
ax = sns.heatmap(data=corr, annot=True, mask=mask)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

population_size = 50
n_generations = 100
GA = Genetic_algorithm(monthly_returns, corr_matrix, risk_free, population_size)

variable = 'return'
GA.run(n_generations,variable)
tree1 = GA.get_tree(variable)
analysis1 = GA.get_analysis()
variable = 'volatility'
GA.run(n_generations,variable)
tree2 = GA.get_tree(variable)
analysis2 = GA.get_analysis()
variable = 'sharpe_ratio'
GA.run(n_generations,variable)
tree3 = GA.get_tree(variable)
analysis3 = GA.get_analysis()

result1_df = tree_to_table(tree1, 'tree_1')
result2_df = tree_to_table(tree2, 'tree_2')
result3_df = tree_to_table(tree3, 'tree_3')
results_df = pd.concat([result1_df, result2_df, result3_df])

results_df.sort_values(by = ['return','volatility'], inplace=True, ascending=[False,True])
b1 = results_df.head(1)
print(results_df)

results_df.sort_values(by = ['volatility','return'], inplace=True, ascending=[True,False])
b2 = results_df.head(1)
print(results_df)

results_df.sort_values(by='sharpe_ratio', inplace=True, ascending=False)
b3 = results_df.head(1)
print(results_df)

plt.figure(figsize=(16,9))
sns.scatterplot(x = results_df['volatility'],y = results_df['return'], size=results_df['generation'], hue=results_df['generation']) #, style=results_df['name'])
plt.scatter(x = b1['volatility'].iloc[0], y = b1['return'].iloc[0], marker = '*', color = 'r', s =500, label = 'Max E[Return]')
plt.scatter(x = b2['volatility'].iloc[0], y = b2['return'].iloc[0], marker = '*', color = 'g', s =500, label = 'Max E[Return]')
plt.scatter(x = b3['volatility'].iloc[0], y = b3['return'].iloc[0], marker = '*', color = 'y', s =500, label = 'Max E[Return]')
plt.show()

best_portfolios = pd.concat([b1,b2,b3])
print(best_portfolios)

w1 = tree2[best_portfolios['generation'].iloc[0]][best_portfolios['chromosome'].iloc[0]].get_weights()
w2 = tree2[best_portfolios['generation'].iloc[1]][best_portfolios['chromosome'].iloc[1]].get_weights()
w3 = tree1[best_portfolios['generation'].iloc[2]][best_portfolios['chromosome'].iloc[2]].get_weights()
d = {'stock_name': stock_names, 'Weights_Portfolio_return_maximum': w1, 'Weights_Portfolio_volatility_minimum': w2, 'Weights_Portfolio_sharperatio_optimal': w3}
best_portfolios_weights = pd.DataFrame(data=d)
print("We therefore got the best weights\n", best_portfolios_weights)
best_portfolios_weights.to_excel('best_portfolios_weights.xlsx')

##

benchmark_prices_df = yf.download(benchmark_ticker, start, end, progress=True, auto_adjust=True)['Close']
benchmark_returns_pct = benchmark_prices_df.pct_change()

benchmark_returns_matrix = pd.DataFrame(benchmark_returns_pct)
benchmark_monthly_return = benchmark_returns_matrix.resample('M').agg(lambda x: (x + 1).prod() - 1)
benchmark_monthly_returns = benchmark_monthly_return.values.flatten()
Date_index = benchmark_monthly_return.index

#Weights_Portfolio_return_maximum
Weights_Portfolio_return_maximum = best_portfolios_weights['Weights_Portfolio_return_maximum'].values

return_max_portfolio_return_matrix = np.dot(monthly_returns.iloc[:, :], Weights_Portfolio_return_maximum)
return_max_portfolio_return = return_max_portfolio_return_matrix.reshape(-1,1).flatten()
return_max_active_returns = return_max_portfolio_return - benchmark_monthly_returns

fig = plt.figure()

plt.plot(Date_index,return_max_active_returns, color='r', label='return_max_active returns')
plt.plot(Date_index,benchmark_monthly_returns, color='b', label='Benchmark')
plt.plot(Date_index,return_max_portfolio_return, color='g', label='return_max_Portfolio')
plt.title(label="Weights_Portfolio_return_maximum - Return")
plt.legend()

fig.show()

return_max_Tracking_error = np.std(return_max_active_returns)

print("return_max_Tracking error is :\n",return_max_Tracking_error)

IR = np.divide(return_max_active_returns,return_max_Tracking_error)

plt.plot(Date_index,IR, color='r', label='Weights_Portfolio_return_maximum - IR')
plt.title(label="return_max_IR")
plt.show()

#Weights_Portfolio_volatility_minimum
Weights_Portfolio_volatility_minimum = best_portfolios_weights['Weights_Portfolio_volatility_minimum'].values

vol_min_portfolio_return_matrix = np.dot(monthly_returns.iloc[:, :], Weights_Portfolio_volatility_minimum)
vol_min_portfolio_return = vol_min_portfolio_return_matrix.reshape(-1,1).flatten()
vol_min_active_returns = vol_min_portfolio_return - benchmark_monthly_returns

fig = plt.figure()

plt.plot(Date_index,vol_min_active_returns, color='r', label='vol_min_active returns')
plt.plot(Date_index,benchmark_monthly_returns, color='b', label='Benchmark')
plt.plot(Date_index,vol_min_portfolio_return, color='g', label='vol_min_Portfolio')
plt.title(label="Weights_Portfolio_volatility_minimum - Return")
plt.legend()

fig.show()

vol_min_Tracking_error = np.std(vol_min_active_returns)
print("vol_min_Tracking error is: \n",vol_min_Tracking_error)

IR = np.divide(vol_min_active_returns,vol_min_Tracking_error)

plt.plot(Date_index,IR, color='r', label='Weights_Portfolio_volatility_minimum - IR')
plt.title(label="Weights_Portfolio_volatility_minimum - IR")
plt.show()


#Weights_Portfolio_sharperatio_optimal
Weights_Portfolio_sharperatio_optimal = best_portfolios_weights['Weights_Portfolio_sharperatio_optimal'].values

sharpe_opt_portfolio_return_matrix = np.dot(monthly_returns.iloc[:, :], Weights_Portfolio_sharperatio_optimal)
sharpe_opt_portfolio_return = sharpe_opt_portfolio_return_matrix.reshape(-1,1).flatten()
sharpe_opt_active_returns = sharpe_opt_portfolio_return - benchmark_monthly_returns

fig = plt.figure()

plt.plot(Date_index,sharpe_opt_active_returns, color='r', label='sharpe_opt_active returns')
plt.plot(Date_index,benchmark_monthly_returns, color='b', label='Benchmark')
plt.plot(Date_index,sharpe_opt_portfolio_return, color='g', label='sharpe_opt_Portfolio')
plt.title(label="Weights_Portfolio_sharperatio_optimal - Return")
plt.legend()

fig.show()

sharpe_opt_Tracking_error = np.std(sharpe_opt_active_returns)
print("sharpe_opt_Tracking error is: \n",sharpe_opt_Tracking_error)

IR = np.divide(sharpe_opt_active_returns,sharpe_opt_Tracking_error)

plt.plot(Date_index,IR, color='r', label='Weights_Portfolio_sharperatio_optimal - IR')
plt.title(label="Weights_Portfolio_sharperatio_optimal - IR")
plt.show()

##predict

y_train = benchmark_monthly_returns[0:60]
##Return _Max Portfolio
x_return_max_train = sm.add_constant(return_max_portfolio_return[0:60])

model_return_max = sm.OLS(y_train,x_return_max_train)
results_return_max = model_return_max.fit()
print(results_return_max.summary())

x_return_max_test = sm.add_constant(return_max_portfolio_return[61:])
y_return_max_test = results_return_max.predict(x_return_max_test)
print(y_return_max_test)

return_max_portfolio_test = return_max_portfolio_return[61:]
IC_return_max = sp.stats.pearsonr(y_return_max_test,return_max_portfolio_test)
print(IC_return_max)

##vol_min Portfolio

print("vol min Portfolio\n")

x_vol_min_train = sm.add_constant(vol_min_portfolio_return[0:60])

model_vol_min = sm.OLS(y_train,x_vol_min_train)
results_vol_min = model_vol_min.fit()
print(results_vol_min.summary())

x_vol_min_test = sm.add_constant(vol_min_portfolio_return[61:])
y_vol_min_test = results_vol_min.predict(x_vol_min_test)
print(y_vol_min_test)

vol_min_portfolio_test = vol_min_portfolio_return[61:]
IC_vol_min = sp.stats.pearsonr(y_vol_min_test,vol_min_portfolio_test)
print(IC_vol_min)

##sharpe_opt Portfolio

print("sharpe_opt Portfolio\n")

x_sharpe_opt_train = sm.add_constant(sharpe_opt_portfolio_return[0:60])

model_sharpe_opt = sm.OLS(y_train,x_sharpe_opt_train)
results_sharpe_opt = model_sharpe_opt.fit()
print(results_sharpe_opt.summary())

x_sharpe_opt_test = sm.add_constant(sharpe_opt_portfolio_return[61:])
y_sharpe_opt_test = results_sharpe_opt.predict(x_sharpe_opt_test)
print(y_sharpe_opt_test)

sharpe_opt_portfolio_test = sharpe_opt_portfolio_return[61:]
IC_sharpe_opt = sp.stats.pearsonr(y_sharpe_opt_test,sharpe_opt_portfolio_test)
print(IC_sharpe_opt)


