import booleanNetwork as bn
import os
import numpy as np

class Reservoir(bn.BooleanNetwork):
    """A Boolean network-based reservoir.

    The reservoir is a combination of an N-node Boolean network and an
    I input nodes. The self.nodes list is indexed such that the first N nodes
    (0 through N-1) are network nodes and the rest (N through N+I-1) are
    input nodes"""

    def __init__(self, numberOfInputs, numberOfConnectionsPerInput,
                 inputFilePath, outputFilePath,
                 numberOfNetworkNodes, linkages, functions, initialNetworkNodeValues):
        """Extends the BooleanNetwork constructor by initializing the input nodes."""

        self.numberOfNetworkNodes = numberOfNetworkNodes
        self.numberOfInputs = numberOfInputs
        self.L = numberOfConnectionsPerInput
        self.inputData = bn.getDataFromFile(inputFilePath)
        super().__init__(numberOfNetworkNodes + numberOfInputs, linkages,
                         functions, initialNetworkNodeValues, outputFilePath)

        for i in range(self.numberOfInputs):
            self.nodes[i + self.numberOfNetworkNodes] = self.inputData[0][i]
        self.outputFilePath = outputFilePath
        self.timestep = 0

        for inputNode in range(self.numberOfInputs):
            hasInput = []
            for _ in range(self.L):
                # select a network node that hasn't already been selected
                # and actually has inputs (i.e. it isn't fixed)
                while True:
                    networkNode = np.random.randint(self.numberOfNetworkNodes)
                    if self.K[networkNode] == 0:
                        continue
                    if networkNode not in hasInput:
                        hasInput.append(networkNode)
                        break
                # find a node in the input set that isn't an input node already
                while True:
                    replacementIndex = np.random.randint(self.K[networkNode])
                    if self.varF[networkNode,replacementIndex] < self.numberOfNetworkNodes:
                        break
                # replace this node with the input node
                self.varF[networkNode][replacementIndex] = self.numberOfNetworkNodes + inputNode

        self.initializeOutput()

    def update(self, iterations=1):
        for iteration in range(iterations):

            for i in range(self.numberOfInputs):
                input = self.inputData[self.timestep][i]
                self.nodes[i + self.numberOfNetworkNodes] = input
                self.networkHistory[self.timestep, i + self.numberOfNetworkNodes] = input

            super().update()
            self.timestep += 1

        self.writeNetworkHistory()

    def initializeOutput(self):
        file = open(self.outputFilePath, 'w')
        stringToWrite = ''
        for i in range(self.numberOfNetworkNodes):
            stringToWrite += 'Node {},'.format(i + 1)
        for i in range(self.numberOfInputs):
            stringToWrite += 'Input Node {},'.format(i + 1)
        # replace last comma with newline
        stringToWrite = stringToWrite[:-1] + '\n'
        file.write(stringToWrite)
        file.close()

    def stateToWrite(self):
        stringToWrite = ''
        data = self.networkHistory # changed from getHistoryAsVectors()
        # note that this change means the regression needs to adjust the input index
        for vector in data:
            for datum in vector:
                stringToWrite += (str(datum) + ',')
            stringToWrite = stringToWrite[:-1] + '\n'

        return stringToWrite

    def getHistoryAsVectors(self, delay=1):
        vectors = []
        for i in range(len(self.networkHistory) - delay):
            vectors.append([])
            for node in self.networkHistory[i + delay][:-self.numberOfInputs]:
                vectors[i].append(node)
            for inputNode in self.networkHistory[i][-self.numberOfInputs:]:
                vectors[i].append(inputNode)
        # remove initial nodes (which don't have a corresponding timestep)
        # (note that I'm not doing this any more since networkUpdatesPerTimestep = 1)
        # from the first vector
        # if len(vectors) > 0:
            # print('before . . .')
            # print(len(vectors))
            # for vec in vectors:
            #     print(len(vec))
            #vectors[0] = vectors[0][5:]
            # print('. . . and after.')
            # print(len(vectors))
            # for vec in vectors:
            #     print(len(vec))
        return vectors

        def setInitialNodeValues(self, values):
            if len(values) != self.numberOfNetworkNodes:
                raise ValueError('Wrong number of inputs.')

            for i in range(len(values)):
                self.nodes[i] = values[i]
                self.networkHistory[-1,i] = values[i]

def addReservoirParameters(numberOfInputs, parameter_getter, *args, **kwargs):
    (varF, F, init) = parameter_getter(*args, **kwargs)

    for _ in range(numberOfInputs):
        zeros = [0] * len(varF[0])
        varF.append(zeros)

        zeros = [0] * len(F[0])
        F.append(zeros)

        init.append(0)

    return (varF, F, init)

directory = '/Users/maxnotarangelo/Documents/ISB/code/BN_realization/'
linkages_filename = 'linkages.csv'
functions_filename = 'functions.csv'
initial_filename = 'initial_nodes.csv'
inputs_filename = 'time_series_data.csv'
outputs_filepath = '/Users/maxnotarangelo/Documents/ISB/code/test_2018-06-26_3.csv'
(varF, F, init) = addReservoirParameters(1, bn.getParametersFromFile, 5, directory + linkages_filename,
                                           directory + functions_filename,
                                           directory + initial_filename)

#r = Reservoir(1, 2, directory + inputs_filename, outputs_filepath, 5, varF, F, init)
# r.update(500)
# r.writeNetworkHistory()
