import numpy as np

import function_arguments as f_utils
import booleanNetwork as bn
from reservoir import Reservoir
from output_layer import OutputLayer

# ensembleSize = 10
N = 50
K = 2.3
I = 1
L = 2
window = 5
delay = 1
dataStreamLength = 100
trainingSize = 20
testSize = 20
O = 10

np.random.seed(0)

functionsToApproximate = []
functionInputs = []
for i in range(O):
    func = f_utils.getRandomBinaryFunction(window, bias=0.5)
    functionsToApproximate.append(func)

    functionInputs.append([(0, x) for x in range(window)])

(varF, F, init) = bn.getRandomParameters(N + I, K, isConstantConnectivity=False)

bn_directory = '/Users/maxnotarangelo/Documents/ISB/code/BN_realization/'
directory = '/Users/maxnotarangelo/Documents/ISB/code/'
res = Reservoir(I, L, bn_directory + 'time_series_data_3.csv', directory + 'test_2018-08-01.csv', N, varF, F, init)

output = OutputLayer(res, O, functionsToApproximate, functionInputs, delay, dataStreamLength)


output.train(trainingSize)
output.test(testSize)
print('\n\n\n\n')
success_rates = output.getSuccessRates()
for rate in success_rates:
    print(rate)
