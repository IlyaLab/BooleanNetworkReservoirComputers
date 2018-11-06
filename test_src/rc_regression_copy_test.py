import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import booleanNetwork as bn
import reservoir
from reservoir import Reservoir

N = 5 # number of network nodes
I = 1 # number of input nodes
L = 2 # number of connections per input

# File paths
input_fp = reservoir.directory + reservoir.inputs_filename
output_fp = reservoir.outputs_filepath

# Network parameters
linkages = bn.varF
functions = bn.F
initialNodes = bn.init

np.random.seed(3)
res = Reservoir(I, L, input_fp, output_fp, N, linkages, functions, initialNodes)
res.getHistoryAsVectors()
res.update(100)
df = pd.DataFrame(res.getHistoryAsVectors())
res.getHistoryAsVectors()

# X_train = np.array(df.iloc[:-20,:-1].sort_index(1))
# X_test = np.array(df.iloc[-20:,:-1].sort_index(1))
# y_train = df.iloc[:-20,-1]
# y_test = df.iloc[-20:,-1]

for i in range(5, 90, 5):
    X_train = df.iloc[:i, :-1]
    X_test = df.iloc[i:, :-1]

    y_train = df.iloc[:i, -1]
    y_test = df.iloc[i:, -1]
    len(y_test)

    model = linear_model.Lasso(alpha=0.001, max_iter=1000)
    model.fit(X_train, y_train)
    print('R^2: ', model.score(X_test, y_test))

# print(model.predict(X_test))
#
# print('R^2: ', model.score(X_test, y_test))
# diff = model.predict(X_test) - y_test
#
# print(diff)
#
# plt.scatter(y_test,model.predict(X_test))
# plt.show()
