import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
import booleanNetwork as bn
import reservoir
from reservoir import Reservoir
import cProfile

ensembleSize = 10
N = 10
K = 2.3
I = 1
L = 3
windowSize = 3
delay = 1
numberOfUpdates = 400
testSize = 100

input_fp = '/Users/maxnotarangelo/Documents/ISB/code/BN_realization/time_series_data.csv'
directory = '/Users/maxnotarangelo/Documents/ISB/code/rc_ensemble_data/'
majority_data_fp = '/Users/maxnotarangelo/Documents/ISB/code/BN_realization/function_data.csv'
majority_data = pd.read_csv(majority_data_fp)

np.random.seed(0)
ensemble = []
print('Beginning initialization.')
for i in range(ensembleSize):
    (links, funcs, inits) = reservoir.addReservoirParameters(I, bn.getRandomParameters,
                                                             N, K, isConstantConnectivity=False)
    res = Reservoir(I, L, input_fp, directory + 'reservoir_%d_output' % i, N,
                    links, funcs, inits)
    res.update(numberOfUpdates)
    # print('Reservoir %d has vector representation\n' % i)
    # print(res.getHistoryAsVectors())
    ensemble.append(res)

    # if i % 5 == 0:
    print(i, '. . .')


y_hat = None
# print('first stage completed.')
success_rates = []
for res in ensemble:
    df = pd.DataFrame(res.networkHistory)
    X = df.iloc[:, :-I]
    X_train = X.iloc[windowSize:-testSize,:]
    # print('len(X_train): ', len(X_train))
    # print('X_train:\n', X_train.iloc[:5,:6])
    X_test = X.iloc[-testSize:,:]
    # print('len(X_test): ', len(X_test))
    # print('X_test:\n', X_test.iloc[:5,:6])

    majority = majority_data.iloc[windowSize - delay:len(X_train) + len(X_test) + windowSize - delay]
    y_train = majority.iloc[:-testSize]
    # print('len(y_train) = ', len(y_train))
    # print('y_train:\n', y_train.iloc[:5])
    y_test = majority.iloc[-testSize:]
    # print('len(y_test) = ', len(y_test))
    # print('y_test:\n', y_test.iloc[:5])

    model = linear_model.Lasso(alpha=0.01, max_iter=10000)
    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    round = lambda x : 1 if x > 0.5 else 0
    rounded = np.array(list(map(round, y_hat)))
    arrayed_y_test = np.array(y_test)
    difference = arrayed_y_test[:,0].T - rounded
    # print(difference)
    success_rate = 1 - sum(np.abs(difference)) / len(difference)
    success_rates.append(success_rate)
    index = len(success_rates) - 1

    print('Network ', index, '\tR^2: ', model.score(X_test, y_test), '\tSuccess rate: ', success_rate)
print('complete')
print(sum(success_rates) / len(success_rates))

dataf = pd.DataFrame(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
print(dataf.head())
