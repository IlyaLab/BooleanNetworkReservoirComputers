import booleanNetwork as bn
from reservoir import Reservoir, addReservoirParameters
import cProfile

N = 5
K = 2
constK = False
I = 1
L = 2
numberOfUpdates = 750


directory = '/Users/maxnotarangelo/Documents/ISB/BN_realization/'
##linkages_filename = 'linkages.csv'
##functions_filename = 'functions.csv'
##initial_filename = 'initial_nodes.csv'
inputs_filename = 'time_series_data.csv'
outputs_filepath = '/Users/maxnotarangelo/Documents/ISB/test_2018-06-27'

def updateReservoir(index):
##    (varF, F, init) = addReservoirParameters(1, bn.getParametersFromFile,
##                                             5, directory + linkages_filename,
##                                             directory + functions_filename,
##                                             directory + initial_filename)
    (varF, F, init) = addReservoirParameters(I, bn.getRandomParameters, N, K,
                                             isConstantConnectivity=constK)

    r = Reservoir(I, L, directory + inputs_filename,
                  outputs_filepath + '_reservoir_%d_output' % index, N, varF, F, init)
    r.update(numberOfUpdates)
    print('done at index %d' % index)

    return r

cProfile.run('updateReservoir(0)')
