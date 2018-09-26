import numpy as np
import matplotlib.pyplot as plt

def distance(vector1,vector2):            #计算两个向量之间的欧式距离
    return np.sqrt(sum(np.power(vector1-vector2, 2)))

def eps_neighbor(vector1,vector2,eps):    #判断
    return distance(vector1,vector2) < eps


def region_query(dataSet, pointId, eps):    #返回指定一个点pointId的邻域内所有的点
    region = []
    numOfVector = np.shape(dataSet)[0]
    for i in range(numOfVector):
        if eps_neighbor(pointId, dataSet[i, :], eps):
            region.append(dataSet[i])
    return np.array(region)

def dbscan(dataSet, eps, minPts):
    numOfVector = np.shape(dataSet)[0]
    clusterId = 0
    clusterAssment = np.zeros((numOfVector,2))     #clusterAssment第一列存储该点属于的聚类Id,第二列存储改点是否被访问过，0：为访问；1：已经访问过


    for i in range(numOfVector):
        if clusterAssment[i, 1] == 0:
            clusterAssment[i, 1] = 1
            regionPoints = region_query(dataSet, dataSet[i, :], eps)
            if np.shape(regionPoints)[0] < minPts:
                clusterAssment[i, 0] = -1
            else:
                clusterId += 1
                clusterAssment[i, 0] = clusterId
                expand_cluster(dataSet, regionPoints, clusterId, clusterAssment, eps, minPts)

                print(clusterId)

    return clusterAssment


def expand_cluster(dataSet, regionPoints, clusterId, clusterAssment, eps, minPts):
    points = regionPoints
    i = 0
    while i < len(points):
        index = np.argwhere(dataSet[:,0] == points[i,0])
        for j in range(np.shape(index)[0]):
            if dataSet[index[j,0],1] == points[i,1]:
                realIndex = index[j,0]


        if clusterAssment[realIndex, 1] == 0:
            clusterAssment[realIndex, 1] =1
            regionPoints_expand = region_query(dataSet, points[i], eps)
            if np.shape(regionPoints_expand)[0] >= minPts:
                points = np.vstack((points, regionPoints_expand))
        if clusterAssment[realIndex, 0] == 0:
            clusterAssment[realIndex, 0] = clusterId

        i += 1

def showPlot(dataSet, clusterAssment):
    numOfVector=np.shape(dataSet)[0]
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numOfVector):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    plt.show()







#主程序

dataSet = []
fileIn = open('data/788points.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split(',')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

dataSet = np.array(dataSet)
clusterAssment = dbscan(dataSet, 2, 15)
showPlot(dataSet,clusterAssment)
#print(clusterAssment)
# regionPoints=region_query(dataSet,dataSet[20],2)
# print(regionPoints[0,1])





