import os
import numpy as np
import booleanNetwork as bn
from reservoir import Reservoir, addReservoirParameters
from output_layer import OutputLayer
from function_arguments import copy, median, parity

ensembleSize = 10
N = 5
K = 1
I = 1
L = 2
delay = 1
dataStreamLength = 10
trainingSize = 20
testSize = 20
O = 1

np.random.seed(0)
bn_directory = os.getcwd() + '/BN_realization/'
directory = os.getcwd() + '/'
print('About to get Boolean network parameters')
(varF, F, init) = bn.getParametersFromFile(N + I,
                                           bn_directory + 'linkages_2.csv',
                                           bn_directory + 'functions_2.csv',
                                           bn_directory + 'initial_nodes_2.csv')
res = Reservoir(I, L, bn_directory + 'time_series_data_3.csv', directory + 'test_2018-07-31.csv', N, varF, F, init)
functionsToApproximate = [copy]
# res = Reservoir(I, L, input_fp, directory + 'reservoir_%d_output' % i, N,
#                   links, funcs, inits)
print('successfully initialized reservoir')

functionInputs = [[(0, 0)]]

output = OutputLayer(res, O, functionsToApproximate, functionInputs, delay, dataStreamLength)
output.train(trainingSize)
output.reservoir

results = output.test(testSize)

#     print(sum(y_test - y_predicted))
#     differences.append(abs(y_test - y_predicted))
#
# for difference in differences:
#     print(difference)
#     print(len(difference) - sum(difference))
#     print(1 - (sum(difference)/len(difference)))

t_input, t_output, t_network, X_data, y_data = output.trainingObjects

# IN = t_input.reshape((trainingSize * output.totalStreamLength))
# OUT = t_output.reshape((trainingSize * dataStreamLength))

in2 = t_input.reshape((trainingSize, output.totalStreamLength))
out2 = t_output.reshape((trainingSize, dataStreamLength))

# print('\n\n\n')
# print(IN)
# for i in range(len(IN)):
#     print(f'[{IN[i]} {OUT[i]}]')
#     if i % dataStreamLength == 0:
#         print()

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
#     total_difference += int(abs(Y_TEST[0,index] - Y_PREDICTED[0,index]))
# success_percentage = 1 - (total_difference / len(Y_TEST[0]))
# print(total_difference)
# print(success_percentage)

# How exactly do the input streams work?
# Do we want a new initialization for the reservoir every time we run an input stream?
# Do we run it with the same initialization? Or do we continue the input stream forever?
