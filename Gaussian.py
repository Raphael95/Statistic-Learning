import csv
import numpy as np

def splitData(dataSet):
    N = np.shape(dataSet)[0]
    diabetesIs = []
    diabetesNo = []
    for i in range(N):
        if dataSet[i, 8] == 0:
            diabetesNo.append(dataSet[i, :])
        else:
            diabetesIs.append(dataSet[i, :])
    return np.array(diabetesIs), np.array(diabetesNo)

def GaussianDist(dataSet,varianList):
    diabetesIs, diabetesNo = splitData(dataSet)
    muIs = np.mean(diabetesIs, axis=0)
    muNo = np.mean(diabetesNo, axis=0)
    sigmaIs = np.var(diabetesIs, axis=0)
    sigmaNo = np.var(diabetesNo, axis=0)
    normalDistIs = []
    normalDistNo = []
    for i in range(len(varianList)):
        result = 1 / (np.sqrt(2 * np.pi * sigmaIs[i])) * np.exp(-((varianList[i] - muIs[i])**2) / (2 * sigmaIs[i]))
        result1 = 1 / (np.sqrt(2 * np.pi * sigmaNo[i])) * np.exp(-((varianList[i] - muNo[i])**2) / (2 * sigmaNo[i]))
        normalDistIs.append(result)
        normalDistNo.append(result1)
    normalDistIs = np.array(normalDistIs)
    normalDistNo = np.array(normalDistNo)
    probNumOfIs = np.shape(diabetesIs)[0] / np.shape(dataSet)[0]
    probNumOfNo = np.shape(diabetesNo)[0] / np.shape(dataSet)[0]
    probIs = 1
    probNo = 1
    for j in range(len(varianList)):
        probIs *= normalDistIs[j]
        probNo *= normalDistNo[j]
    probIs = probIs * probNumOfIs
    probNo = probNo * probNumOfNo

    if probIs > probNo:
        print(probIs)
        print('该体检者有患糖尿病的概率很大')
    else:
        print(probNo)
        print('该体检者没有糖尿病')





dataSet=[]
fileIn = open('data/diabetes.csv')
for line in fileIn.readlines():
    lineArr = line.strip().split(',')
    dataSet.append([float(lineArr[0]), float(lineArr[1]),float(lineArr[2]),float(lineArr[3]),float(lineArr[4]),float(lineArr[5]),float(lineArr[6]),float(lineArr[7]),float(lineArr[8])])

dataSet = np.array(dataSet)

varianList = [5, 117, 92, 0, 0, 34.1, 0.337, 38]
GaussianDist(dataSet, varianList)
print(dataSet)

