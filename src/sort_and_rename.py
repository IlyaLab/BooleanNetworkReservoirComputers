import os
import sys
import datetime

now = datetime.datetime.now()
now_formatted = 'T'.join(str(now).split(' '))

job_name = sys.argv[1]

directory = job_name + '_' + now_formatted

if len(sys.argv) > 2:
    filepath = sys.argv[2]
else:
    filepath = os.getcwd()

for filename in os.listdir(os.getcwd()):
    if filename.startswith(job_name):
        read_file = open(filename)
        metadata = []
        lines = read_file.readlines()
        for line in lines:
            print(line)
            if line == '\n':
                break
            metadata.append(line)

        parameters = []
        for metadatum in metadata:
            metadatum_list = metadatum.split(' ')
            rewritten_metadatum_list = []
            print(f'metadatum_list = {metadatum_list}')
            for item in metadatum_list:
                item = item.rstrip('\n')
                if item == '=':
                    item = '-'
                rewritten_metadatum_list.append(item)

            parameter = ''.join(rewritten_metadatum_list)
            print(f'parameter = {parameter}')
            parameters.append(parameter)
        new_filename = '_'.join(parameters) + '.csv'

        read_file.close()

        data = lines[len(metadata) + 1:]
        write_file = open(filename, 'w')
        for line in data:
            write_file.write(line)
        write_file.close()

        os.renames(filename, os.path.join(filepath, directory, new_filename))
