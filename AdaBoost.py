import numpy as np

def predictMat(dataSet, dim, threshold, inequal):
    predictValue = np.ones((np.shape(dataSet)[0], 1))
    if inequal == 'lt':
        predictValue[dataSet[:, dim] <= threshold] = -1
    else:
        predictValue[dataSet[:, dim] > threshold] = -1
    return predictValue

def weakClassifty(x, dim, threshold, inequal):
    if inequal == 'lt':
        if x <= threshold:
            return -1
        else:
            return 1
    else:
        if x <= threshold:
            return 1
        else:
            return -1

def buildStump(dataSet, weightMat):
    m, n = np.shape(dataSet)
    weightMat = weightMat.T
    unequal = ['lt', 'rt']
    minError = 1000.0
    bestStump ={}
    for i in range(n-1):
        fvalue = dataSet[:, i]
        feature = set(fvalue)
        for j in feature:
            for inequal in unequal:
                predictValue  = predictMat(dataSet, i, j, inequal)
                predictValue = predictValue.T
                errorMat = np.ones((1, m))
                errorMat[predictValue == dataSet[:, n-1]] = 0
                error = np.array(np.dot(errorMat, weightMat.T))
                print('真实的error值是: ' + str(error))
                if error[0][0] < minError:
                    minError = error[0][0]
                    bestClass = predictValue.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = j
                    bestStump['inequal'] = inequal
    return bestStump, minError, bestClass

def adaBoost(dataSet, numIt):
    weakClass = []
    m, n = np.shape(dataSet)
    weightValue = np.ones((m, 1))/m
    aggClass = np.zeros((1, m))
    for i in range(numIt):
        bestStump, minError, bestClass = buildStump(dataSet, weightValue)
        alpha = float((1 / 2) * np.log((1 - minError) / minError))
        bestStump['alpha'] = alpha
        weakClass.append(bestStump)
        expon = np.multiply(-1 * alpha * dataSet[:, n-1], bestClass)
        weightValue = np.multiply(weightValue, np.exp(expon))
        weightValue = weightValue / weightValue.sum()
        aggClass += alpha * bestClass
        aggError = np.multiply(np.sign(aggClass) != np.mat(dataSet[:, n-1]), np.ones((m, 1)))
        aggErrorRate = aggError.sum() / m
        if aggErrorRate == 0.0:
            break
    return weakClass

def  testAdaBoost(dataSet, weakClass):
    m, n = np.shape(dataSet)[0]
    aggClass = np.zeros((m, 1))
    for i in range(len(weakClass)):
        bestClass = predictMat(dataSet, weakClass[i]['dim'], weakClass[i]['threshold'], weakClass[i]['inequal'])
        aggClass += bestClass * weakClass[i]['alpha']
    return np.sign(aggClass)






dataSet =[]
fileIn = open('data/AdaBoost.txt')
for arr in fileIn.readlines():
    arrLine = arr.strip().split('\t')
    dataSet.append([float(arrLine[0]), float(arrLine[1]), float(arrLine[2])])

dataSet = np.array(dataSet)
weightMat = np.array([[0.2],
             [0.2],
             [0.2],
             [0.2],
             [0.2]])
# bestStump, minError, bestClass = buildStump(dataSet, weightMat)
# print(0.22354321 * bestClass)

weakClass = adaBoost(dataSet, 9)

print(weakClass)