# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import datasets, linear_model
import booleanNetwork as bn
from reservoir import Reservoir, addReservoirParameters
import cProfile

ensembleSize = 3
N = 500
K = 2.3
constK = False
I = 1
L = 5
windowSize = 3
numberOfUpdates = 750
testSize = 100

input_fp = '/Users/maxnotarangelo/Documents/ISB/BN_realization/time_series_data.csv'
directory = '/Users/maxnotarangelo/Documents/ISB/rc_ensemble_data/'


def createAndUpdateReservoir(index):
    (varF, F, init) = addReservoirParameters(I, bn.getRandomParameters, N, K, isConstantConnectivity=constK)
    res = Reservoir(I, L, input_fp, directory + 'reservoir_%d_output' % index, N,
                    varF, F, init)
    res.update(numberOfUpdates)
    # print('Reservoir %d has vector representation\n' % i)
    # print(res.getHistoryAsVectors())
    return res

    # if i % 5 == 0:
    print(index, '. . .')

print('begin profiling . . .')
cProfile.run('createAndUpdateReservoir(0)')
