
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
N_list = [100] # [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
K = 2.0
I = 1
L_list = [50]  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#L_list = [ 30 ] 
window = 3
delay = 1
dataStreamLength = 10
trainingSize = 150
testSize = 150
O = 2 ** (2 ** window) #10
seed = 123

#np.random.seed(seed)

# Initialize functions and inputs
functionInputs = [f_utils.getInputTuple(window) for _ in range(O)]
functionsToApproximate = []
for i in range(O):
    function_vector = f_utils.convertIntToBinaryVector(i, 2 ** window)
    func = f_utils.convertVectorToFunction(function_vector)
    functionsToApproximate.append(func)

# Test parameter set
data = np.zeros((len(N_list), len(L_list)))
print("instance,<K>,N,L,function,success_rate" )

for i in range(len(N_list)):
    for j in range(len(L_list)):

        L = L_list[j] * N_list[i] // 100
       

        # Initialize reservoir
        bn_directory = os.getcwd() + '/../BN_realization/'
        directory = os.getcwd() + '/../output/'
        varF, F, init = bn.getRandomParameters(N_list[i] + I, K + (L / N_list[i]), isConstantConnectivity=False)

        #print ( varF )
        #avgK = 0.0 
        #for row in varF :
        #    mystring = ''
        #    for col in row : 
        #       if col != -1 :
        #          avgK  += 1  
        #       mystring += '{0:3d}'.format(col)
        #    print( mystring )
        #print( [ avgK ,  len( varF ),  avgK / len( varF)  ] )   

        res = Reservoir(I, L, bn_directory + 'time_series_data_3.csv',
                        directory + 'experiment1_2018-10-10.csv', N_list[i], varF, F, init)

        # Train and test output layer
        output = OutputLayer(res, O, functionsToApproximate,
                             functionInputs, delay, dataStreamLength)
        output.train(trainingSize)
        output.test(testSize)

        # add results to data
        data[i, j] = sum([output.successRates[k] for k in range(O)]) / O
        
        #for k in range(O) :
        #     #"{0:}".format(seed,i,j,K,output.successRates[k])
        #    print("{0:d},{1:.3f},{2:d},{3:d},{4:d},{5:.5f}".format( seed,K,N_list[i],L_list[j],k,output.successRates[k] ))

time = int(time.clock() - start)
# Write metadata to file
f_utils.printParameters(N_list, K, I, L_list, window, delay, dataStreamLength,
                        trainingSize, testSize, O, 'allThreeBit', seed, time)

# write data to file
df = pd.DataFrame(data)
df.index = N_list
print(df.to_csv(header=L_list))
