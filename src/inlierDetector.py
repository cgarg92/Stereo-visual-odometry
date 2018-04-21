import numpy as np

def findClique(d3dPointsT1, d3dPointsT2, distDifference):
    # in-lier detection algorithm
    numPoints = d3dPointsT1.shape[0]
    W = np.zeros((numPoints, numPoints))

    # diff of pairwise euclidean distance between same points in T1 and T2
    for i in range(numPoints):
        for j in range(numPoints):
            T2Dist = np.linalg.norm(d3dPointsT2[i,:] - d3dPointsT2[j,:])
            T1Dist = np.linalg.norm(d3dPointsT1[i,:] - d3dPointsT1[j,:])
            if (abs(T2Dist - T1Dist) < distDifference):
                W[i, j] = 1

    count = 0
    maxn = 0
    maxc = 0

    # Find point with maximum degree and store in maxn
    for i in range(numPoints):
        for j in range(numPoints):
            if W[i,j] == 1:
                count = count+1
        if count > maxc:
            maxc = count
            maxn = i
        count=0

    clique = [maxn]
    isin = True

    while True:
        potentialnodes = list()
        # Find potential nodes which are connected to all nodes in the clique
        for i in range(numPoints):
            for j in range(len(clique)):
                isin = isin & bool(W[i, clique[j]])
            if isin == True and i not in clique:
                potentialnodes.append(i)
            isin=True

        count = 0
        maxn = 0
        maxc = 0
        # Find the node which is connected to the maximum number of potential nodes and store in maxn
        for i in range(len(potentialnodes)):
            for j in range(len(potentialnodes)):
                if W[potentialnodes[i], potentialnodes[j]] == 1:
                    count = count+1
            if count > maxc:
                maxc = count
                maxn = potentialnodes[i]
            count = 0
        if maxc == 0:
            break
        clique.append(maxn)
    return clique
