import os
import sys
import time
import numpy as np
import pandas as pd
import booleanNetwork as bn
from reservoir import Reservoir
from output_layer import OutputLayer
import function_arguments as f_utils

start = time.clock()
N_list = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
K = 2
I = 1
L_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
window = 3
delay = 1
dataStreamLength = 10
trainingSize = 150
testSize = 150
O = 1
seed = 0  # sys.argv[1]

np.random.seed(seed)

# Initialize functions and inputs
functionInputs = [[(0, 0), (0, 1), (-1, 1)]]
functionsToApproximate = [f_utils.median]

# Test parameter set
data = np.zeros((len(N_list), len(L_list)))
for i in range(len(N_list)):
    for j in range(len(L_list)):
        L = L_list[j] * N_list[i] // 100

        # Initialize reservoir
        bn_directory = os.getcwd() + '/BN_realization/'
        directory = os.getcwd() + '/'
        varF, F, init = bn.getRandomParameters(N_list[i] + I, K + (L / N_list[i]), isConstantConnectivity=False)
        res = Reservoir(I, L, bn_directory + 'time_series_data_3.csv',
                        directory + 'experiment1_2018-08-03.csv', N_list[i], varF, F, init)

        # Train and test output layer
        output = OutputLayer(res, O, functionsToApproximate, functionInputs,
                             delay, dataStreamLength,
                             nonRecursiveArgs=[[(0, 2)]])
        output.train(trainingSize)
        output.test(testSize)

        # add results to data
        if j % 5 == 0:
            print('adding data')
        data[i, j] = sum([output.successRates[k] for k in range(O)]) / O

time = int(time.clock() - start)
# Write metadata to file
f_utils.printParameters(N_list, K, I, L_list, window, delay, dataStreamLength,
                        trainingSize, testSize, O, 'recursiveMedian', seed, time)

# write data to file
df = pd.DataFrame(data)
df.index = N_list
print(df.to_csv(header=L_list))
