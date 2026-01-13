import numpy as np
import operator as op


def createDataSet():
    dataSet = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return dataSet, labels


def classify0(inX, dataSet, labels, k):
    # 计算inX与dataSet中各个点的距离
    dataSetSize = dataSet.shape[0]
    diffValue = dataSet - np.tile(inX, (dataSetSize, 1))
    s1 = diffValue ** 2
    s2 = s1.sum(axis=1)
    s3 = s2 ** 0.5
    # 返回前k个距离最近的点的序号
    numbers = s3.argsort()
    # 统计k个点对应的标签出现的次数
    classCount = {}
    for i in range(k):
        votedLable = labels[numbers[i]]
        classCount[votedLable] = classCount.get(votedLable, 0) + 1
    # 找到出现次数最的标签
    sortedClassCount = sorted(
        classCount.items(), key=op.itemgetter(1), reverse=True)
    # 返回该标签及其数量
    # returnStr = "label: " + \
    #    sortedClassCount[0][0] + " Count: " + str(sortedClassCount[0][1])
    returnStr = f'label: {sortedClassCount[0][0]} Count: {sortedClassCount[0][1]}'
    return returnStr, classCount
    # return sortedClassCount[0][0]
