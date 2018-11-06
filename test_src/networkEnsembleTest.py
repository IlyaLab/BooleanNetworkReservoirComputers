import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import booleanNetwork as bn

def stepsUntilAttractorIsReached(df):
    for index in df.index:
        for testIndex in df.iloc[index+1:,:].index:
            # print('%d, %d' % (index, testIndex))
            # print(df.iloc[index] - df.iloc[testIndex])
            if not (df.iloc[index] - df.iloc[testIndex]).any():
                return index
    return None

N = 10
K = 2
p = 0.5
ensemble_size = 1000
networkUpdates = 11
loc = '/Users/maxnotarangelo/Documents/ISB/BN_testing'

ensemble = []
for index in range(ensemble_size):
    (linkages, functions, init) = bn.getRandomParameters(
            N, K, isConstantConnectivity=True, bias=p)
    net = bn.BooleanNetwork(N, linkages, functions, init,
                            loc, 'network_%d.csv' % index)
    ensemble.append(net)

print('Done initializing ensemble')

steps_until_attractor = []
for i in range(len(ensemble)):
    ensemble[i].update(networkUpdates)
    filepath = os.path.join(loc, 'network_%d.csv' % i)
    net_data = pd.read_csv(filepath)
    steps_until_attractor.append(stepsUntilAttractorIsReached(net_data))
    if i % 100 == 0:
        print(i)

s = pd.Series(steps_until_attractor)
plt.hist(s, align='right')
plt.show()
s.value_counts()
s.value_counts().index
s.value_counts()

X = []
for i in s.value_counts().index:
    X.append(i)

X

sns.regplot(np.array(X), s.value_counts(), fit_reg=False)

s.value_counts().iat[11] / sum(s.value_counts())
