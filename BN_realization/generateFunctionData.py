

# memoryLength should be even so that the majority function is always defined;
# the input to the majority function includes the current timestep and the
# previous memoryLength timesteps
def generateMajorityFunctionData(memoryLength,
                                 timeseries_filepath,
                                 function_output_filepath):
    time_series_file = open(timeseries_filepath)
    contents = time_series_file.read()
    time_series = []
    for char in contents:
        if char == '1':
            time_series.append(1)
        elif char == '0':
            time_series.append(0)
    time_series_file.close()

    majority_data = [-1 for _ in range(memoryLength)]
    for i in range(memoryLength, len(time_series)):
        numberOfOnes = sum(time_series[i - memoryLength: i + 1])
        if 2 * numberOfOnes > memoryLength:
            majority_data.append(1)
        else:
            majority_data.append(0)

    # Note that right now this only works with 1 input
    strToWrite = 'Majority\n'
    for datum in majority_data:
        strToWrite += str(datum)
        strToWrite += '\n'
    output_file = open(function_output_filepath, 'w')
    output_file.write(strToWrite)


folder = '/Users/maxnotarangelo/Documents/ISB/BN_realization/'
ts_filepath = folder + 'time_series_data.csv'
output_filepath = folder + 'function_data.csv'

generateMajorityFunctionData(2, ts_filepath, output_filepath)
