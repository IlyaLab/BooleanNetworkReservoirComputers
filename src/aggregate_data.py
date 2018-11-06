import os
import sys
import numpy as np
import pandas as pd

filename = sys.argv[1]

if len(sys.argv) > 2:
    data_folder = sys.argv[2]
else:
    data_folder = os.getcwd()

sum_df = None
for file in os.listdir(data_folder):
    if sum_df is None:
        sum_df = pd.read_csv(file)
    else:
        sum_df = sum_df + pd.read_csv(file)

ave_df = sum_df / len(os.listdir(data_folder))

filepath = os.path.join(data_folder, filename + '_aggregated.csv')
ave_df.to_csv(filepath)
