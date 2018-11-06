import random

for j in range(15):
    nodes = []

    for i in range(6):
        nodes.append(random.randint(0,1))

    N = len(nodes)

    K = 1
    linkages = []
    for i in range(N):
        linkages.append(random.randrange(0,N))

    timenodes = []
    timenodes.append(nodes)


    for i in range(10):
        timenodes.append([])
        for j in range(N):
            # All of the functions are the 'copy' function
            timenodes[-1].append(timenodes[-2][linkages[j]])

    for timestep in timenodes:
        print(timestep)

    print('Next network:')
