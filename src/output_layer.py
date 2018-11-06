from copy import copy

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LassoCV

import booleanNetwork as bn
from reservoir import Reservoir, addReservoirParameters
from function_arguments import generateFunctionDataFromFunction

class OutputLayer(object):

    def __init__(self, reservoir, numberOfOutputNodes, functions,
                 inputsToFunctions, delay, dataStreamLength, nonRecursiveArgs=[[]]):
        # print('Beginning __init__')
        n = numberOfOutputNodes
        if len(functions) != n or len(inputsToFunctions) != n:
            raise ValueError('Function parameter(s) do not match number of output nodes.')

        self.reservoir = reservoir
        self.inputData = np.array(self.reservoir.inputData, dtype=np.int8)
        if self.inputData.ndim == 1:
            self.inputData.shape = (len(self.inputData), 1)
        self.numberOfOutputNodes = numberOfOutputNodes
        self.functions = functions
        self.inputsToFunctions = inputsToFunctions
        self.delay = delay
        self.dataStreamLength = dataStreamLength

        if nonRecursiveArgs == [[]]:
            self.nonRecursiveArgs = [[] for _ in range(numberOfOutputNodes)]
        else:
            self.nonRecursiveArgs = nonRecursiveArgs

        self.numberOfInputsToFunctions = max(len(x) for x in inputsToFunctions)
        self.windows = np.zeros(self.numberOfOutputNodes, dtype=int)
        for i in range(self.numberOfOutputNodes):
            self.windows[i] = 1 + max(t for (n, t) in inputsToFunctions[i])
            if len(self.nonRecursiveArgs[i]) > 0: # fix this
                nr_window = 1 + max(t for (n, t) in nonRecursiveArgs[i])
                self.windows[i] = max(self.windows[i], nr_window)
        self.window = max(self.windows)
        # TODO: check the next two lines
        self.shift = self.window - 1 + self.delay  # I don't know if delay should be here
        self.totalStreamLength = self.dataStreamLength + self.shift
        self.inputDataIndex = 0
        self.successRates = []

    def generateData(self, numberOfDataStreams):
        if numberOfDataStreams * self.totalStreamLength \
                        * self.reservoir.numberOfInputs > \
                        len(self.inputData) - self.inputDataIndex:
            raise ValueError('Not enough input data to complete training/testing.')

        # Set the values of all of the training inputs from the random input streams
        trainingInput = np.full((numberOfDataStreams, self.totalStreamLength,
                                self.reservoir.numberOfInputs), -1, dtype=np.int8)
        for i in range(numberOfDataStreams):
            for j in range(self.totalStreamLength):
                    # this is a 1-D array with length self.reservoir.numberOfInputs
                    trainingInput[i,j] = self.inputData[i*self.totalStreamLength+j + self.inputDataIndex]
        self.inputDataIndex += numberOfDataStreams * self.totalStreamLength * self.reservoir.numberOfInputs
        if -1 in trainingInput:
            raise ValueError('The the full training input was not initialized.')
        # print('Input set trained.')

        # Generate function data
        trainingOutput = np.full((self.numberOfOutputNodes, numberOfDataStreams,
                                  self.dataStreamLength), -1, dtype=np.int8)
        for i in range(self.numberOfOutputNodes):
            for j in range(numberOfDataStreams):
                functionData = generateFunctionDataFromFunction(
                        trainingInput[j], self.windows[i],
                        self.inputsToFunctions[i], self.functions[i],
                        self.dataStreamLength,
                        nonRecursiveArgs=self.nonRecursiveArgs[i])
                for k in range(len(functionData)):
                    trainingOutput[i,j,k] = functionData[k]
        if -1 in trainingOutput:
            raise ValueError('The the full training output was not initialized.')
        # print('Output set trained.')

        # generate Boolean network data
        trainingNetwork = np.full((numberOfDataStreams, self.totalStreamLength,
                                    self.reservoir.numberOfNetworkNodes),
                                   -1, dtype=np.int8)
        for i in range(numberOfDataStreams):
            r_temp = copy(self.reservoir)  # make sure this works and that I don't need to use deepcopy

            initialState = bn.getRandomInitialNodeValues(self.reservoir.numberOfNetworkNodes)
            r_temp.setInitialNodeValues(initialState)
            r_temp.inputData = trainingInput[i]
            r_temp.update(self.totalStreamLength - 1) # - 1 since we include the initial state
            trainingNetwork[i] = r_temp.networkHistory[:,:-self.reservoir.numberOfInputs]

        # print('Network data gathered.')

        # generate X matrix
        X_data = np.full((numberOfDataStreams * self.dataStreamLength,
                           self.reservoir.numberOfNetworkNodes),
                          -1, dtype=np.int8) # no type
        for i in range(len(trainingNetwork)):
            X_data[i * self.dataStreamLength:(i+1) * self.dataStreamLength] = trainingNetwork[i,self.shift:]

        # Generate list of y-vectors
        y_data = np.full((self.numberOfOutputNodes, numberOfDataStreams *
                           self.dataStreamLength), -1, dtype=np.int8)  # no type
        for i in range(self.numberOfOutputNodes):
            for j in range(numberOfDataStreams):
                y_data[i,j*self.dataStreamLength:(j+1) * self.dataStreamLength] = trainingOutput[i,j]

        return (X_data, y_data)

    def train(self, trainingSize):
        # print('Training sequence initiated.')
        X_train, y_train = self.generateData(trainingSize)
        self.X_train = X_train

        self.models = []
        for i in range(self.numberOfOutputNodes):
            reg = LassoCV(cv=5, max_iter=10000)
            self.models.append(reg)
            self.models[i].fit(X_train, y_train[i])

    def test(self, testSize):
        # print('Testing sequence initiated.')
        X_test, y_test = self.generateData(testSize)

        shifted_signum = lambda x : 1 if x > 0.5 else 0
        y_predicted_raw = np.full((self.numberOfOutputNodes, testSize * self.dataStreamLength), -1.0)
        y_predicted = np.full((self.numberOfOutputNodes, testSize * self.dataStreamLength), -1, dtype=np.int8)
        for i in range(self.numberOfOutputNodes):
            y_predicted_raw[i] = self.models[i].predict(X_test)
            y_predicted[i] = np.array(list(map(shifted_signum, y_predicted_raw[i])))
            differenceVector = [abs(y_test[i,j] - y_predicted[i,j]) for j in range(testSize * self.dataStreamLength)]
            successRate = 1 - (sum(differenceVector) / (testSize * self.dataStreamLength))
            self.successRates.append(successRate)
        #y_predicted_raw = np.matmul(X_test, self.models[0].coef_)

        # print('number of output nodes: %d' % self.numberOfOutputNodes)
        # print(f'model weights: {self.models[0].coef_}')
        # print('X_test,  y_test:,  y_predicted_raw, y_predicted')
        # for i in range(len(X_test)):
        #     print(f'{X_test[i]}  {y_test[0,i]}  {y_predicted_raw[0,i]}  {y_predicted[0,i]}')
        # print('\n\n\n\n\n')
        return (y_test, y_predicted)

    def getSuccessRates(self):
        return self.successRates

# Make sure to fix filepaths
