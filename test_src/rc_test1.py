import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns
import booleanNetwork as bn
from reservoir import Reservoir
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

fp = '/Users/maxnotarangelo/Documents/ISB/BN_realization/time_series_data.csv'
N = 100
numberOfInputs = 1
L = 5
(random_linkages, random_functions, random_init) = bn.getRandomParameters(N, 2)
r = Reservoir(numberOfInputs, L, fp, N, random_linkages,
              random_functions, random_init)
r.update(100)


reservoir_data_file = '/Users/maxnotarangelo/Documents/ISB/log.csv'
rdf = pd.read_csv(reservoir_data_file)
rdf.head()

func_data_file = '/Users/maxnotarangelo/Documents/ISB/BN_realization/function_data.csv'
fdf = pd.read_csv(func_data_file)
fdf.head()

rdf['Majority'] = fdf.get('Majority')

X = rdf.drop(['Input Node 1', 'Majority'], axis=1)

lm = LinearRegression()
lm.fit(X, rdf['Majority'])
lm.predict(X)[:30]

plt.scatter(rdf['Majority'], lm.predict(X))
plt.xlabel('Actual majority values $y(t)$')
plt.ylabel('Predicted majority values $\hat{y}(t)$')
plt.show()

# pd.DataFrame(zip(X.columns, lm.coef_), columns=['features', 'estimatedCoefficients'])
# print(results.head())
print(X.shape)
