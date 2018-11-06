import os
import sys
import numpy as np
import booleanNetwork as bn
from reservoir import Reservoir
from output_layer import OutputLayer
import function_arguments as f_utils

N = 500
K = 2
I = 1
L = 10 * 20  # int(sys.argv[1])
window = 3
delay = 1
dataStreamLength = 10
trainingSize = 150
testSize = 150
O = 2 ** (2 ** window)

seed = 0
np.random.seed(seed)

bn_directory = os.getcwd() + '/BN_realization/'
directory = os.getcwd() + '/'
print(os.getcwd())

varF, F, init = bn.getRandomParameters(N + I, K, isConstantConnectivity=False)
res = Reservoir(I, L, bn_directory + 'time_series_data_3.csv',
                directory + 'test_2018-08-03.csv', N, varF, F, init)
print('reservoir initialized')
functionInputs = [f_utils.getInputTuple(window) for i in range(O)]

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

print(f'N = {N}')
print(f'K = {K}')
print(f'I = {I}')
print(f'L = {output.reservoir.L}')
print(f'window = {window}')
print(f'delay = {delay}')
print(f'dataStreamLength = {dataStreamLength}')
print(f'trainingSize = {trainingSize}')
print(f'testSize = {testSize}')
print(f'seed = {seed}')
print(f'O = {O}')
print()

print('Function,Accuracy')
for i in range(O):
    print(f'{i},{output.successRates[i]}')
