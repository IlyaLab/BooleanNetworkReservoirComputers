import os
import sys
import numpy as np
import booleanNetwork as bn
from reservoir import Reservoir
from output_layer import OutputLayer
import function_arguments as f_utils

N = 50
K = 2.5
I = 1
L = 5
window = 3
delay = 1
dataStreamLength = 10
trainingSize = 150
testSize = 150
O = 2 ** (2 ** window)

seed = int(sys.argv[1])
np.random.seed(seed)

bn_directory = os.getcwd() + '/BN_realization/'
directory = os.getcwd() + '/'

varF, F, init = bn.getRandomParameters(N + I, K, isConstantConnectivity=False)
res = Reservoir(I, L, bn_directory + 'time_series_data_3.csv',
                directory + 'test_2018-08-03.csv', N, varF, F, init)
functionInputs = [[(0, 0), (0, 1), (0, 2)] for i in range(O)]

functionsToApproximate, functionVectors = [], []
for i in range(O):
    function_vector = f_utils.convertIntToBinaryVector(i, 2 ** window)
    func = f_utils.convertVectorToFunction(function_vector)
    functionsToApproximate.append(func)
    # Only for printing purposes
    functionVectors.append(function_vector)

output = OutputLayer(res, O, functionsToApproximate,
                     functionInputs, delay, dataStreamLength)
output.train(trainingSize)
output.test(testSize)

print(f'Seed: {seed}')
print()

for i in range(O):
    print(f'{functionVectors[i]}:  {output.successRates[i]}')
