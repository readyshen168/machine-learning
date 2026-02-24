'''
香农决策树
'''
from math import log


def calShannonSimple(dataSet, normalize=False):
    '''
    输入含有特征比例的列表，计算熵值

    :param dataSet: 
    :param normalize: 是否对列表中的数据进行归一化处理
    '''
    # 检查列表中各项的取值范围
    if not dataSet:
        raise ValueError("输入列表不能为空")
    for ratio in dataSet:
        if ratio < 0:
            raise ValueError(f"概率值必须非负， 但得到{ratio}")
    # 检查列表中各项之和是否接近1， 否则进行归一化处理（根据参数normalize）
    total = sum(dataSet)
    if not abs(total - 1.0) < 1e-9:
        if normalize:
            # 自动归一化
            dataSet = [ratio/total for ratio in dataSet]
        else:
            raise ValueError(f"概率和必须为1， 但得到{total}")

    shannonValue = 0.0
    for ratio in dataSet:
        shannonValue -= ratio * log(ratio, 2)

    print(f'{dataSet} entropy: {shannonValue}')
    return shannonValue


def calcShannonEnt2(data_set):
    '''第二种计算数据集标签熵值的方法'''
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 提取所有的特征比例组成一个列表
    labelProbs = []
    for label, count in label_counts.items():
        labelProb = float(count)/num_entries
        labelProbs.append(labelProb)
    # 调用calShannonSimple, 传入特征比例列表
    calShannonSimple(labelProbs)


def calcShannonEnt(data_set):
    '''
    calcShannonEnt 的 Docstring

    :param dataSet: 说明
    '''
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for label, count in label_counts.items():
        prob = float(count)/num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def createDataSet():
    '''
    createDataSet 的 Docstring
    '''
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']

    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def splitDataSet(dataSet, axis, value) -> list[list[any]]:
    """
    按照给定的特征划分数据集

    :param dataSet: 待划分的数据集
    :type dataSet: list[list]

    :param axis: 划分数据集的特征
    :type axis: int

    :param value: 划分数据集的特征值
    :type value: int

    retDataSet 用来储存划分后数据集的列表
    循环数据集中的每一行 存在变量featVec中
        如果该行数据对应的特征位置的值为value:
            列表reducedFeatVec储存该特征位置前的数据
            列表reducedFeatVec加入该特征位置后的数据
            retDataSet加入reducedFeatVec列表
    返回retDataSet列表

    :return retDataSet: list[list]
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式（选择能让信息增益最大的特征）

    :param dataSet: 待划分的数据集
    :type dataSet: list[list]

    numFeatures 特征数量
    baseEntropy 数据集的原始香农值
    bestInfoGain = 0.0 最佳信息增益值，初始为0
    bestFeature = -1 带来最佳信息增益的特征，初始为-1

    循环每一个特征：
        # 得出该特征的取值
        featList 该列表储存本特征的所有取值，从dataSet的每行数据列表exmaple中获取
        uniqueVals 获取featList中的唯一属性值的集合
        newEntropy = 0.0 以该特征划分的数据集的熵值，初始为0

        遍历该特征下的每个属性值：
            subDataSet 使用splitDataset划分出来的子数据集
            prob 该子数据集占所有数据集的比例
            newEntropy 通过calcShannonEnt计算该子数据集的熵值, 并累加得出该特征划分的数据集的熵值
        infoGain 信息增益 = 原始熵增 - 以该特征划分数据集后的熵值
        如果 信息增益大于最佳增益：
            最佳信息增益 = 该信息增益
            最佳特征 = 该特征

    返回最佳特征

    """


def majorityCnt(classList):
    """
    对类别数据集进行统计，返回出现最多的类别

    :param classList: 类别的数据集
    :type classList: list[list]

    """
    # 声明一个对象classCount用来储存类别对应的数量
    # 遍历classList，统计各个类别的数量
    # 如果classList中的某个类别vote不在classCount中，则该类别的数量记为0
    # 否则自增1
    # 对classCount中的元素采用sorted方法进行排序，以元素的取值为key, 从大到小逆序
    # 返回上面排序后的列表sortedClassCount的第一个元素的第一个值，即为classList中数量最多的类别
