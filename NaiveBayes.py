import numpy as np


def calcuParameter(dataSet):       #使用了拉普拉斯平滑
    parameter = 1
    N = np.shape(dataSet)[0]
    K = 1
    S1 = 1
    S2 = 1

    exitstData = []
    exitstData1 = []
    exitstData2 = []
    for i in range(N):
        if dataSet[i, 2] != dataSet[0, 2] and dataSet[i, 2] not in exitstData:
            exitstData.append(dataSet[i, 2])
            K += 1
        if dataSet[i, 0] != dataSet[0, 0] and dataSet[i, 0] not in exitstData1:
            exitstData1.append(dataSet[i, 0])
            S1 += 1
        if dataSet[i, 1] != dataSet[0, 1] and dataSet[i, 1] not in exitstData2:
            exitstData2.append(dataSet[i, 1])
            S2 += 1

    return parameter, N, K, S1, S2

def prioriProb(dataSet):
    parameter , N, K, S1, S2= calcuParameter(dataSet)
    label1 = '感冒'
    label2 = '过敏'
    label3 = '脑震荡'
    numOfLabel1 = 0
    numOfLable2 = 0
    numOfLabel3 = 0
    label1List = []
    label2List = []
    label3List = []
    for i in range(N):
        if dataSet[i, 2] == label1:
            numOfLabel1 += 1
            label1List.append(i)
        if dataSet[i, 2] == label2:
            numOfLable2 += 1
            label2List.append(i)
        if dataSet[i, 2] == label3:
            numOfLabel3 += 1
            label3List.append(i)
    denominator = N + K * parameter
    probL1 = (numOfLabel1 + parameter)/denominator
    probL2 = (numOfLable2 + parameter)/denominator
    probL3 = (numOfLabel3 + parameter)/denominator
    return probL1, probL2, probL3, numOfLabel1, numOfLable2, numOfLabel3, label1List, label2List, label3List

def conditionProb(dataSet, condition, condition1):
    parameter, N, K, S1, S2 = calcuParameter(dataSet)
    probL1, probL2, probL3, numOfLabel1, numOfLabel2, numOfLabel3, label1List, label2List, label3List = prioriProb(dataSet)
    print('probL3 = '+str(probL3))
    result = np.zeros((K, 2))
    for i in range(K):
        numOfCondition = 0
        numOfCondition1 = 0
        bayesProb = 0
        if i == 0:
            for j in label1List:
                if dataSet[j, 0] == condition:
                    numOfCondition += 1
                if dataSet[j, 1] == condition1:
                    numOfCondition1 += 1
            bayesProb = probL1 * ((numOfCondition + parameter)/(numOfLabel1 + S1)) * ((numOfCondition1 + parameter) /
                                                                                      (numOfLabel1 + S2))
        if i == 1:
            for j in label2List:
                if dataSet[j, 0] == condition:
                    numOfCondition += 1
                if dataSet[j, 1] == condition1:
                    numOfCondition1 += 1
            bayesProb = probL2 * ((numOfCondition + parameter) / (numOfLabel2 + S1)) * ((numOfCondition1 + parameter) /
                                                                                        (numOfLabel2 + S2))
        if i == 2:
            for j in label3List:
                if dataSet[j, 0] == condition:
                    numOfCondition += 1
                if dataSet[j, 1] == condition1:
                    numOfCondition1 += 1
            bayesProb = probL3 * ((numOfCondition + parameter) / (numOfLabel3 + S1)) * ((numOfCondition1 + parameter) /
                                                                                        (numOfLabel3 + S2))

        result[i, 0] = i
        result[i, 1] = bayesProb

    return result



dataSet = []
fileIn = open('data/diaster.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([str(lineArr[0]), str(lineArr[1]), str(lineArr[2])])

dataSet = np.array(dataSet)
result = conditionProb(dataSet, '打喷嚏', '农夫')
print(result)
# parameter, N, K, S1, S2 = calcuParameter(dataSet)
# pro1, pro2, pro3, a1, a2, a3, l1, l2, l3 = prioriProb(dataSet)
#
# print(dataSet)
# print(K,S1,S2)
# print(pro1, pro2, pro3)