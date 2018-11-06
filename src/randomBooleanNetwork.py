import os
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from booleanNetwork import BooleanNetwork


class RandomBooleanNetwork(BooleanNetwork):
    """Representation of a Boolean network, generated randomly."""

    def __init__(self, numberOfNodes, connectivity,
                 isConstantConnectivity=True, bias=0.5):
        if numberOfNodes < connectivity:
            raise ValueError('Connectivity larger than number of nodes')

        # Generate connectivities
        connectivities = []
        if (isConstantConnectivity):
            for _ in range(numberOfNodes):
                connectivities.append(connectivity)
            maxConnectivity = connectivity
        else:
            # Note that this is a specific distribution (uniform)
            # of the values of K[i]
            for _ in range(numberOfNodes):
                connectivities.append(0)
            for _ in range(int(connectivity * numberOfNodes)):
                connectivities[random.randrange(0, numberOfNodes)] += 1

            maxConnectivity = max(connectivities)

        # Generate linkages
        linkages = []
        for i in range(numberOfNodes):
            linkages.append([])
            for _ in range(connectivities[i]):
                while True:
                    newnode = random.randrange(0, numberOfNodes)
                    if newnode not in linkages[i]:
                        linkages[i].append(newnode)
                        break

            for _ in range(maxConnectivity - connectivities[i]):
                linkages[i].append(-1)

        # Generate functions
        functions = []
        for i in range(numberOfNodes):
            functions.append([])
            # Initialize a maxK by N matrix filled with -1
            for _ in range(2 ** maxConnectivity):
                functions[i].append(-1)
            # Fill in K[i] values
            for j in range(2 ** connectivities[i]):
                if random.random() < bias:
                    functions[i][j] = 1
                else:
                    functions[i][j] = 0

        # Initialize nodes
        initialNodeValues = []
        for _ in range(numberOfNodes):
            initialNodeValues.append(random.randint(0, 1))

        super().__init(numberOfNodes, linkages, functions, initialNodeValues)


print('Hello world.')
net = RandomBooleanNetwork(5, 2, isConstantConnectivity=False)
(f, vf) = net.getRealization()
print(f)
print(vf)

print("The initial values of the network are")
print(net.nodes)
net.update(10)
print('network updated')

df = pd.read_csv(os.path.join(os.getcwd(), 'log.txt'))
df.head()
# plt.show()
# sns.lmplot()
