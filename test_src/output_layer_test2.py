import os
import numpy as np
import booleanNetwork as bn
from reservoir import Reservoir, addReservoirParameters
from output_layer import OutputLayer
import function_arguments as f_utils
from function_arguments import copy, median, parity

ensembleSize = 10
N = 500
K = 2
I = 1
L = 400
window = 5
delay = 1
dataStreamLength = 20
trainingSize = 150
testSize = 150
O = 1
outputWindow = 0

np.random.seed(12)
bn_directory = '/Users/maxnotarangelo/Documents/ISB/code/BN_realization/'
directory = '/Users/maxnotarangelo/Documents/ISB/code/'
varF, F, init = bn.getRandomParameters(N + I, K, isConstantConnectivity=True)


print(f'F = {F}')
print(f'init = {init}')

res = Reservoir(I, L, bn_directory + 'time_series_data_4.csv', directory + 'test_2018-08-03_parity.csv', N, varF, F, init)
print('successfully initialized reservoir')

function_vector = [0, 1, 1, 0, 1, 0, 0, 1]
# functionsToApproximate = [f_utils.convertVectorToFunction(function_vector)] # this is the parity function in vector form
functionsToApproximate = [parity]



functionInputs = [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]]

output = OutputLayer(res, O, functionsToApproximate, functionInputs, delay, dataStreamLength) # , nonRecursiveArgs=[[(0,2)]]
output.train(trainingSize)
output.reservoir

results = output.test(testSize)
# differences = []
# for y_test, y_predicted in results:
#     print(sum(y_test - y_predicted))
#     differences.append(abs(y_test - y_predicted))
#
# for difference in differences:
#     print(difference)
#     print(len(difference) - sum(difference))
#     print(1 - (sum(difference)/len(difference)))


# IN = t_input.reshape((trainingSize * output.totalStreamLength))
# OUT = t_output.reshape((trainingSize * dataStreamLength))

# in2 = t_input.reshape((trainingSize, output.totalStreamLength))
# out2 = t_output.reshape((trainingSize, dataStreamLength))

# print('\n\n\n')
# print(IN)
# for i in range(len(IN)):
#     print(f'[{IN[i]} {OUT[i]}]')
#     if i % dataStreamLength == 0:
#         print()

# for i in range(trainingSize):
#     for j in range(dataStreamLength):
#         print(f'[{in2[i,j]}] [{out2[i,j]}]')
#     for k in range(output.totalStreamLength - dataStreamLength):
#         print(f'[{in2[i,dataStreamLength+k]}] [-]')
#     print()


print('\n\n\n\n\n')
Y_TEST, Y_PREDICTED = results
print(Y_TEST[0])
print(Y_PREDICTED[0])
total_difference = 0
for index in range(len(Y_TEST[0])):
    total_difference += int(abs(Y_TEST[0,index] - Y_PREDICTED[0,index]))
success_percentage = 1 - (total_difference / len(Y_TEST[0]))
print(total_difference)
print(success_percentage)
print(output.successRates)
print(output.models[0].alpha)
print(output.models[0].max_iter)

# How exactly do the input streams work?
# Do we want a new initialization for the reservoir every time we run an input stream?
# Do we run it with the same initialization? Or do we continue the input stream forever?
