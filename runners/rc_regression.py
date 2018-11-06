import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
import booleanNetwork as bn
import reservoir
from reservoir import Reservoir

N = 5 # number of network nodes
I = 1 # number of input nodes
L = 2 # number of connections per input
K = 2 # number of connections per network node
U = 5 # number of network updates per input timestep

# File paths
input_fp = reservoir.directory + reservoir.inputs_filename
output_fp = reservoir.outputs_filepath

# Network parameters
linkages = bn.varF
functions = bn.F
initialNodes = bn.init

res = Reservoir(I, L, input_fp, output_fp, U, N, linkages, functions, initialNodes)
res.update(100)
df = pd.DataFrame(res.getHistoryAsVectors())
