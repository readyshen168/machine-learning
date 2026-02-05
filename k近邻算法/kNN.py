"""
module kNN
"""
import operator as op
import numpy as np


# 生成数据集和标签
def createDataSet():
    """
    createDataset
    """
    dataSet = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # np.array...测试一下此处的array
    print(type(dataSet))  # <class 'numpy.ndarray'>
    print(dataSet.shape)  # (4,2) - NumPy数组才有shape属性
    labels = ['A', 'A', 'B', 'B']
    return dataSet, labels

# k近邻算法分类器


def classify0(inX, dataSet, labels, k):
    """
    classify0
    """
    # dataSet有多少行
    dataSetSize = dataSet.shape[0]
    # 求差值
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 求平方
    sqDiffMat = diffMat**2
    # 沿每行进行求和 axis=1
    # sqDistances = np.sum(sqDiffMat, 1)
    sqDistances = sqDiffMat.sum(axis=1)
    # 求开根得距离
    distances = sqDistances**0.5
    # 选出距离最小的前k名 相当于在表中对各行进行排序,返回主序号的集合
    sortedDistIndicies = distances.argsort()
    # 储存标签投票数的字典
    classCount = {}
    # 循环k次，统计标签票数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对字典内的标签进行降序排序
    sortedClassCount = sorted(classCount.items(),
                              key=op.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]
