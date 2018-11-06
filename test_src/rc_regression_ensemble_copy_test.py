import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
import booleanNetwork as bn
import reservoir
from reservoir import Reservoir

ensembleSize = 50
N = 10
K = 2
I = 1
L = 5

input_fp = '/Users/maxnotarangelo/Documents/ISB/BN_realization/time_series_data.csv'
directory = '/Users/maxnotarangelo/Documents/ISB/rc_ensemble_data/'

np.random.seed(0)
ensemble = []
for i in range(ensembleSize):
    (links, funcs, inits) = bn.getRandomParameters(N, K)
    res = Reservoir(I, L, input_fp, directory + 'reservoir_%d_output' % i, N,
                    links, funcs, inits)
    res.update(100)
    # print('Reservoir %d has vector representation\n' % i)
    # print(res.getHistoryAsVectors())
    ensemble.append(res)
    if i % 5 == 0:
        print(i, '. . .')

success = 0
failure = 0
data_real = None
print('first stage completed.')
for res in ensemble:
    df = pd.DataFrame(res.getHistoryAsVectors())

    # print('df:\n', df)
    X_train = df.iloc[:-20, :-1]
    X_test = df.iloc[-20:, :-1]

    y_train = df.iloc[:-20, -1]
    y_test = df.iloc[-20:, -1]

    # print(X_train)
    model = linear_model.Lasso(alpha=0.001, max_iter=1000)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > 0.9:
        success += 1
    else:
        failure += 1
    if not type(data_real) == pd.Series:
        data_real = y_test
        data_predicted = pd.Series(model.predict(X_test))
    else:
        data_real = data_real.append(y_test)
        data_predicted = data_predicted.append(pd.Series(model.predict(X_test)))
print('complete')
print('success percentage: ', float(success) / (success + failure))
plt.scatter(data_real, data_predicted)
plt.show()
