import numpy as np
import booleanNetwork as bn
from reservoir import Reservoir, addReservoirParameters
from output_layer import OutputLayer
from function_arguments import median, parity

ensembleSize = 10
N = 5
K = 2.3
I = 2
L = 1
window = 3
delay = 1
dataStreamLength = 10
trainingSize = 20
testSize = 20
O = 2
outputWindow = 0

np.random.seed(0)
bn_directory = '/Users/maxnotarangelo/Documents/ISB/code/BN_realization/'
directory = '/Users/maxnotarangelo/Documents/ISB/code/'
(varF, F, init) = bn.getRandomParameters(N + I, K, isConstantConnectivity=False)
res = Reservoir(I, L, bn_directory + 'time_series_data_2.csv', directory + 'test_2018-07-26.csv', N, varF, F, init)
functionsToApproximate = [median, parity]
# res = Reservoir(I, L, input_fp, directory + 'reservoir_%d_output' % i, N,
#                   links, funcs, inits)
print('successfully initialized reservoir')

functionInputs = [[(0, 0), (0, 1), (0, 2)], [(0, 0), (0, 1), (1, 0), (1, 1)]]

output = OutputLayer(res, O, functionsToApproximate, functionInputs, delay, dataStreamLength)
output.train(trainingSize)
output.reservoir
a, b, c, d, e = output.trainingObjects
print(a)
print(len(a))
print(len(a[0]))
print()
print(b)
print()
print(c)

print(len(c))
print(len(c[0]))
print(len(c[0,0]))
print(output.shift)

print('\n\n')
print('X_train =')
print(d)
print(len(d))
print(len(d[0]))

print('\n\n')
print(len(b))
print(len(b[0]))
print(len(b[0,0]))

print(e)
print(len(e))
print(len(e[0]))
results = output.test(testSize)

y_tests, y_predicts = results

differences = []
for i in range(len(y_tests)):
    difference = []
    for j in range(len(y_tests[i])):
        difference.append(abs(y_tests[i,j] - y_predicts[i,j]))
    differences.append(difference)

success_rates = []
for i in range(len(y_tests)):
    success_rates.append(1 - (sum(differences[i]) / len(y_tests[i])))

print(success_rates)

print(output.successRates)

# How exactly do the input streams work?
# Do we want a new initialization for the reservoir every time we run an input stream?
# Do we run it with the same initialization? Or do we continue the input stream forever?
