
# coding: utf-8

from numpy import *
import numpy as np
import operator

def createDataSet():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0], 
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # 计算arrary的行数和列数，第一个是行数
    # 距离计算
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.interitems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
