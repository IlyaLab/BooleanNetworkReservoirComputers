import sys

def create_timeseries_csv(input_filepath, output_filepath, numberOfInputs):
    input_file = open(input_filepath)
    raw_input = input_file.readlines()
    input_file.close()

    one_line_input = ''
    for line in raw_input:
        one_line_input += line

    list_input = []
    for char in one_line_input:
        if char == '1':
            list_input.append(1)
        elif char == '0':
            list_input.append(0)

    strToWrite = ''
    for i in range(numberOfInputs):
        strToWrite += 'Input_%d,' % (i + 1)
    strToWrite = strToWrite[:-1] + '\n'

    for i in range(len(list_input) // numberOfInputs):
        for j in range(numberOfInputs):
            strToWrite += str(list_input[numberOfInputs * i + j])
            strToWrite += ','
        strToWrite = strToWrite[:-1] + '\n'

    output_file = open(output_filepath, 'w')
    output_file.write(strToWrite)


inputFile = '/Users/maxnotarangelo/Documents/ISB/code/BN_realization/' + \
                sys.argv[1] # 'time_series_raw_2.txt'
outputFile = '/Users/maxnotarangelo/Documents/ISB/code/BN_realization/' + \
                sys.argv[2] # 'time_series_data_3.csv'

create_timeseries_csv(inputFile, outputFile, int(sys.argv[3])) 
