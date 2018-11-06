import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

all_three_bit = r'~/Documents/ISB/data/allThreeBit_2018-08-08T14:05:27.306720/allThreeBit_aggregated.csv'
recursive_median = r'/Users/maxnotarangelo/Documents/ISB/data/recursive_median_2018-08-08T14:27:04.735517/recursiveMedian_aggregated.csv'
recursive_parity = r'/Users/maxnotarangelo/Documents/ISB/data/recursive_parity_2018-08-08T12:06:12.381248/recursiveParity_aggregated.csv'
filepath = recursive_parity

df = pd.read_csv(filepath)
df

df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
df.rename(index=lambda x : 10*x + 10, inplace=True)
df

df.rename(index={60:100, 70:200, 80:300, 90:400, 100:500}, inplace=True)
df

graph = sns.heatmap(df.iloc[::-1])
graph.set_title('Mean Accuracy of Recursive Parity')
graph.set(xlabel='Percentage of Nodes Connected to the Input Node', ylabel='Number of Nodes')
plt.show()
