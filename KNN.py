import numpy as np
import operator


def KNN_classify(dataSet, label, inputData, k):
    m, n = np.shape(dataSet)
    diffMat = np.tile(inputData, (m, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = np.sum(sqDiffMat, axis=1)
    distance = sqDistance ** 0.5
    sortedIndex = distance.argsort()
    sortedIndex = list(sortedIndex)

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedIndex.index(i)]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


dataSet = [[1.0, 1.1],
           [1.0, 1.0],
           [0, 0],
           [0, 0.1]]
label = ['A', 'A', 'B', 'B']
inputData = [[1.2, 1.0]]
result = KNN_classify(dataSet, label, inputData, 3)
print(result)
