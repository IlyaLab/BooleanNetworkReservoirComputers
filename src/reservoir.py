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

        numberOfKEqualsZeroNodes = 0
        for i in range(self.numberOfNetworkNodes):
            if self.K[i] == 0:
                numberOfKEqualsZeroNodes += 1

        for inputNode in range(self.numberOfInputs):
            hasInput = []
            for i in range(self.L):
                # select a network node that hasn't already been selected
                # and actually has inputs (i.e. it isn't fixed)

                # Not sure if this is the best solution
                # if i >= self.numberOfNetworkNodes - numberOfKEqualsZeroNodes:
                #     # raise ValueError('not enough nodes with K > 0')
                #     break

                # create a list of network nodes (hasInput) that we're going to connect the input node to;
                # ignore network nodes with no inputs
                while True:
                    networkNode = np.random.randint(self.numberOfNetworkNodes)

                    if networkNode not in hasInput:
                        hasInput.append(networkNode)
                        if self.K[networkNode] == 0:
                            # added to make L close to N still work; this might slightly skew results
                            # also, probably not robust to changes in I,
                            # and definitely not robust to changes in bias,
                            # since it is set at bias = 0.5
                            self.K[networkNode] = 1
                            self.F[networkNode,0], self.F[networkNode,1] = np.random.randint(2), np.random.randint(2)
                        break
                # print('made it through the first while loop at index %d' % i)
                # find a node in the input set that isn't an input node already
                # print(f'in the second while loop, L = {self.L}, \
                # numberOfNetworkNodes = {self.numberOfNetworkNodes}, \
                # creating the {i}th connection')
                replacementIndex = np.random.randint(self.K[networkNode]) # adding one here should fix the issue, but I have no idea why it's worked up until now
                # print('number of network nodes: %d' % self.numberOfNetworkNodes)
                # print(f'self.varF =\n{self.varF}')
                # print(f'networkNode = {networkNode}')
                # print(f'len(varF[networkNode] = {len(self.varF[networkNode])}')
                # print(f'replacementIndex = {replacementIndex}, self.K[{networkNode}] = {self.K[networkNode]}, self.varF[{networkNode}, {replacementIndex}] = {self.varF[networkNode,replacementIndex]}, self.numberOfNetworkNodes = {self.numberOfNetworkNodes}')
                # replace this node with the input node
                # print('made it through the second while loop at index %d' % i)
                # I changed the [][] to [,] - make sure this works
                self.varF[networkNode,replacementIndex] = self.numberOfNetworkNodes + inputNode

        # print('input connections initialized')
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

directory = os.getcwd() + '/BN_realization/'
linkages_filename = 'linkages.csv'
functions_filename = 'functions.csv'
initial_filename = 'initial_nodes.csv'
inputs_filename = 'time_series_data.csv'
outputs_filepath = os.getcwd() + '/test_2018-08-02_1.csv'
# (varF, F, init) = addReservoirParameters(1, bn.getParametersFromFile, 5, directory + linkages_filename,
#                                            directory + functions_filename,
#                                            directory + initial_filename)

#r = Reservoir(1, 2, directory + inputs_filename, outputs_filepath, 5, varF, F, init)
# r.update(500)
# r.writeNetworkHistory()
