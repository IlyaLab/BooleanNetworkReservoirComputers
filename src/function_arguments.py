import numpy as np
from inspect import signature

def copy(arg):
    if arg == 1:
        return 1
    elif arg == 0:
        return 0
    else:
        raise ValueError('input to the copy function was not "0" or "1"')

def median(*args):
    """Returns the median/majority of a set of {0, 1} inputs.

    Parameters
    ----------
    *args : {0, 1}
        The inputs to the median function.

    Returns
    -------
    1 : int
        returns 1 if the majority of inputs are 1
    2 : int
        returns 0 if the majority of inputs are 0

    Raises
    ------
    ValueError
        When there are an even number of inputs.

    """
    if len(args) % 2 == 0:
        raise ValueError("median function must have an odd number of inputs.")
    for a in args:
        if 2 * sum(args) > len(args):
            return 1
        else:
            return 0

def parity(*args):
    return sum(args) % 2


inputData = np.array([[1, 1], [0, 1], [0, 1], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 1], [0, 1], [0, 0]])

# functionData = []
# for i in range(len(inputData)):
#     if i < 3:
#         continue
#     functionData.append(median(inputData[i-1], inputData[i-2], inputData[i-3]))
# functionData

def generateFunctionDataFromFunction(inputData, window, inputsToFunction, function, dataStreamLength, nonRecursiveArgs=[]):
    """Add useful docstring here

    """
    shift = window - 1

    functionData = []
    for i in range(len(inputData)):
        if i < shift:
            continue
        elif len(functionData) >= dataStreamLength:
            break;
        args = []
        functionArgsIndex = 0
        for (nodeIndex, timeIndex) in inputsToFunction:
            if nodeIndex == -1:
                if i - shift < timeIndex: # need to test this
                    nr_nodeIndex, nr_timeIndex = nonRecursiveArgs[functionArgsIndex]
                    args.append(inputData[i - nr_timeIndex, nr_nodeIndex])
                    functionArgsIndex += 1
                else:
                    args.append(functionData[i - shift - timeIndex])
            else:
                args.append(inputData[i - timeIndex, nodeIndex])
        args = tuple(args)
        functionData.append(function(*args))

    return functionData

def generateTruthTable(function, numberOfArgs):
    """

    Parameters
    ----------
    function :
        param numberOfArgs:
    numberOfArgs :


    Returns
    -------

    """
    #sig = signature(function)
    #numberOfArgs = len(sig.parameters)
    tableSize = 2 ** numberOfArgs

    truthTable = []
    for i in range(tableSize):
        functionValue = function(*convertIntToBinaryTuple(i, numberOfArgs))
        truthTable.append(functionValue)
    return np.array(truthTable, dtype=np.int8)

def convertVectorToFunction(vector):
    for i in range(len(vector)):
        divisor = 2 ** i
        if divisor >= len(vector):
            break
        if len(vector) % divisor != 0:
            raise ValueError('The length of the vector is not a power of 2')

    def func(*args):
        input = convertBinaryTupleToInt(args)
        # figure out outputs from inputs here
        return vector[input]

    return func

def convertIntToBinaryVector(integer, vector_length):
    tup = convertIntToBinaryTuple(integer, vector_length)
    return list(tup)

def convertIntToBinaryTuple(integer, tuple_length):
    """

    Parameters
    ----------
    integer :
        param tuple_length:
    tuple_length :


    Returns
    -------

    """
    if (integer >= 2 ** tuple_length):
        raise ValueError('integer is too large to be represented in tuple_length bits')
    vec = []
    for index in range(tuple_length - 1, -1, -1):
        bit = 1 if 2 ** index & integer else 0
        vec.append(bit)

    binary_tuple = tuple(vec)
    return binary_tuple

def convertBinaryTupleToInt(binary_tuple):
    integer = 0
    for index in range(len(binary_tuple)):
        integer += 2 ** (len(binary_tuple) - 1 - index) * binary_tuple[index]

    return integer

def getRandomBinaryFunction(numberOfInputs, bias=0.5):
    functionVector = []
    for _ in range(2 ** numberOfInputs):
        rand = np.random.random()
        if rand >= bias:
            functionVector.append(1)
        else:
            functionVector.append(0)

    return convertVectorToFunction(functionVector)

def getInputTuple(window):
    inputList = []
    for i in range(window):
        inputList.append((0, i))

    return inputList

def printParameters(N, K, I, L, window, delay, dataStreamLength,
                    trainingSize, testSize, O, functions, seed, time):
    print(f'functions = {functions}')
    print(f'N = {N}')
    print(f'K = {K}')
    print(f'I = {I}')
    print(f'L = {L}')
    print(f'window = {window}')
    print(f'delay = {delay}')
    print(f'dataStreamLength = {dataStreamLength}')
    print(f'trainingSize = {trainingSize}')
    print(f'testSize = {testSize}')
    print(f'O = {O}')
    print(f'seed = {seed}')
    print(f'time = {time}')
    print()

func_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
generateFunctionDataFromFunction(inputData, 2, func_inputs, parity, 10)
inputData

convertIntToBinaryTuple(5, 4)

convertBinaryTupleToInt((1, 0, 0, 1))

generateTruthTable(parity, 5)

f = convertVectorToFunction([1, 0, 0, 1, 0, 0, 1, 1])
f(0, 0, 0)

parity(1, 1, 1)

median(1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0)

getInputTuple(5)
