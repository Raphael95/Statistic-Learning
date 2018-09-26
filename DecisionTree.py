import numpy as np
import math

def createDataSet():
    dataSet = [['sunny', 'hot', 'high', 'false', 'N'],
               ['sunny', 'hot', 'high', 'true', 'N'],
               ['overcast', 'hot', 'high', 'false', 'Y'],
               ['rain', 'cool', 'normal', 'false', 'Y'],
               ['overcast', 'cool', 'normal', 'true', 'Y'],
               ['sunny', 'mild', 'high', 'false', 'N'],
               ['sunny', 'mild', 'normal', 'false', 'Y'],
               ['rain', 'mild', 'normal', 'false', 'Y'],
               ['sunny', 'mild', 'normal', 'true', 'Y'],
               ['overcast', 'mild', 'high', 'true', 'Y'],
               ['rain', 'mild', 'high', 'true', 'N'],
               ['rain', 'cool', 'normal', 'true', 'N'],
               ['rain', 'mild', 'high', 'false', 'Y']]

    dataSet = np.array(dataSet)
    labels = ['outlook','temperature','humidity','windy','play']
    return dataSet, labels

def calShannonEnt(dataSet):         #计算熵
    label = {}
    for item in dataSet:
        if item[-1] not in label.keys():
            label[item[-1]] = 1
        else:
            label[item[-1]] += 1

    shannonEnt = 0.0
    numEntries = len(dataSet)
    for key in label:
        prob = float(label[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def splitData(dataSet, axis, value):  #根据给定的属性值筛选数据集
    data = []
    for item in dataSet:
        if item[axis] == value:
            itemSlice = list(item[:axis])
            itemSlice.extend(item[axis+1:])
            data.append(itemSlice)
    return np.array(data)

def bestFeature(dataSet):
    numEntries =len(dataSet)
    featureNum = np.shape(dataSet)[1] - 1
    baseShannonEnt = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeatureId = -1
    for i in range(featureNum):
        lists = [item[i] for item in dataSet]
        featureList = set(lists)
        featureShannonEnt = 0.0
        for fea in featureList:
            subDataSet = splitData(dataSet, i, fea)
            prob = float(len(subDataSet)) / numEntries
            featureShannonEnt += prob * calShannonEnt(subDataSet)

        infoGain =baseShannonEnt - featureShannonEnt
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatureId = i

    return bestFeatureId

def createTree(dataSet, labels):
    classList = [item[-1] for item in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    bestFeatureId = bestFeature(dataSet)
    features = labels[bestFeatureId]
    myTree = {features: {}}
    classFeature = [value[bestFeatureId] for value in dataSet]
    uniqueClassFeature = set(classFeature)
    del(labels[bestFeatureId])
    for v in uniqueClassFeature:
        subLabel = labels[:]
        myTree[features][v] = createTree(splitData(dataSet, bestFeatureId, v), subLabel)
    return myTree

def classify(inputTree, inputLabel, testData):
    firstNode = list(inputTree.keys())[0]
    second_dic = inputTree[firstNode]
    featureIndex = inputLabel.index(firstNode)
    for key in second_dic.keys():
        if testData[featureIndex] == key:
            if isinstance(second_dic[key], dict):
                classType = classify(second_dic[key], inputLabel, testData)
            else:
                classType = second_dic[key]
    return classType


dataSet, label = createDataSet()
print(dataSet)
myTree = createTree(dataSet, label)
label = ['outlook', 'temperature', 'humidity', 'windy', 'play']
test = ['rain', 'mild', 'normal', 'true']
classLabel = classify(myTree, label, test)
print(myTree)
print(classLabel)