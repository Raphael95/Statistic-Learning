import numpy as np
import matplotlib.pyplot as plt

def distance(vector1, vector2):                             #计算两个向量间的距离
    return np.sqrt(sum(np.power(vector1-vector2, 2)))

def createCenterPoints(dataSet, k):                         #随机生成K个点作为初始质心
    numOfVector, dimention = np.shape(dataSet)
    centerPoints = np.zeros((k, dimention))

    for i in range(k):
        randomNumber = np.random.randint(0, numOfVector)
        centerPoints[i, :] = dataSet[randomNumber, :]

    return centerPoints

def kmeans(dataSet, k):
    numOfVector=np.shape(dataSet)[0]
    clusterAssment=np.mat(np.zeros((numOfVector, 2)))         #clusterAssment矩阵中第一列存储的是聚类的索引，第二列存储的是改点到聚类质心的距离
    clusterChanged=True

    centerPoints=createCenterPoints(dataSet, k)

    while clusterChanged:
        clusterChanged=False

        for i in range(numOfVector):
            minDistance = 100000.0
            minIndex = 0
            for j in range(k):
                dis = distance(dataSet[i, :], centerPoints[j, :])
                if dis < minDistance:
                    minDistance = dis
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, 0] = minIndex
                clusterAssment[i, 1] = minDistance

        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centerPoints[j, :] = np.mean(pointsInCluster, axis=0)

    print('cluster complete!')
    return clusterAssment, centerPoints

def showPlot(dataSet, k, clusterAssment, centerPoints):

    numOfVector, dimention=np.shape(dataSet)

    if dimention != 2:
        print('sorry! i can not draw, because dimention of your data is not 2')
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']

    if k > len(mark):
        print
        "Sorry! Your k is too large! please contact Raphael"
        return 1

    for i in range(numOfVector):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']

    for i in range(k):
        plt.plot(centerPoints[i, 0], centerPoints[i, 1], mark[i])

    plt.show()




print('step1: load the data...')

dataSet = []
fileIn = open('data/data.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])


print('step2: clustering')
print(dataSet)

dataSet = np.array(dataSet)
k = 4

clusterAssment, centerPoints=kmeans(dataSet, k)


print('step3: show the result')
showPlot(dataSet, k, clusterAssment, centerPoints)








