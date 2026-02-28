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
    计算data_set最后一列(类标签)的香农熵
    calcShannonEnt 的 Docstring

    :param dataSet: 说明
    '''
    # num_entries变量存储data_set的行数，也就是待计算熵值的类标签的数量
    num_entries = len(data_set)
    # label_counts对象存储列标签对应的数量
    label_counts = {}
    # 遍历data_set中的每行数据feat_vec
    for feat_vec in data_set:
        # 通过feat_vec提取当前标签current_label
        current_label = feat_vec[-1]
        # 如果当前标签不存在于label_counts中
        if current_label not in label_counts:
            # 则当前标签的值记为0
            label_counts[current_label] = 0
        # 如果当前标签存在于label_counts对象中,则其值自增1
        label_counts[current_label] += 1
    # 熵的初始值shannon_ent记为0.0
    shannon_ent = 0.0
    # 遍历label_counts对象，提取标签label和标签对应的值count
    for label, count in label_counts.items():
        # 计算该标签在总数量中的占比prob
        prob = float(count)/num_entries
        # 计算该标签的熵值shannon_ent: prob*log(prob,2)
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

    """
    # numFeatures 特征数量
    numFeatures = len(dataSet[0]) - 1
    # baseEntropy 数据集的原始香农值
    baseEntropy = calcShannonEnt(dataSet)
    # bestInfoGain = 0.0 最佳信息增益值，初始为0
    bestInfoGain = 0.0
    # bestFeature = -1 带来最佳信息增益的特征，初始为-1
    bestFeature = -1

    # 循环每一个特征：
    for axis in range(numFeatures):
      # 1.得出该特征的取值集合（去重）
        # featList 该列表储存本特征的所有取值，从dataSet的每行数据列表exmaple中获取
        featList = [example[axis] for example in dataSet]
        # uniqueVals 获取featList中的唯一属性值的集合
        uniqueVals = set(featList)
        # newEntropy = 0.0 以该特征划分的数据集的熵值，初始为0
        newEntropy = 0.0

      # 2.得出以该特征划分数据集后的熵值
        # 遍历该特征下的每个属性值：
        for value in uniqueVals:
            # subDataSet 使用splitDataset划分出来的子数据集
            subDataSet = splitDataSet(dataSet, axis, value)
            # prob 该子数据集占所有数据集的比例
            prob = len(subDataSet)/(float)(len(dataSet))
            # newEntropy 通过calcShannonEnt计算该子数据集的熵值, 并累加得出该特征划分的数据集的熵值
            newEntropy += prob * calcShannonEnt(subDataSet)

      # 3.更新带来最佳增益的特征
        # infoGain 信息增益 = 原始熵增 - 以该特征划分数据集后的熵值
        infoGain = baseEntropy - newEntropy
        # 如果信息增益大于最佳增益：
        if infoGain > bestInfoGain:
            # 最佳信息增益 = 该信息增益
            bestInfoGain = infoGain
            # 最佳特征 = 该特征
            bestFeature = axis

    # 返回最佳特征
    return bestFeature


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


def createTree(dataSet, labels):
    """
    用递归的方式构建决策树
    :param dataSet: 特征值和类别值的数据集
    :type dataSet: list[list]

    :param labels: 特征名称的数据集
    :type labels: list[list]

    """
    # 新建子标签subLabels，复制labels参数引用，以免更改原标签列表(后面涉及对该列表元素的删除操作）
    subLabels = labels[:]
# 迭代的结束条件，返回具体类别
    # 从数据集获取类别的值列表classList
    classList = [example[-1] for example in dataSet]
    # 如果classList里的类别值只有一种，则返回该类别值
    if classList.count(classList) == len(classList):
        return classList[0]
    # 如果类别值不止一种且特征已分完，也就是dataSet只有一列了，则返回最多的类别值（调用函数majorityCnt）
    if dataSet[0] == 1:
        return majorityCnt(classList)

# 构建树
  # 1 找最佳特征
    # 1-1 用函数chooseBestFeatureToSplit找到信息增益最大的特征bestFeat
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 1-2 找出bestFeat对应labels标签集中的类别名bestFeatLabel
    bestFeatLabel = subLabels[bestFeat]
    # 1-3 以对象的方式存储树myTree,以bestFeatLabel作属性，其对应的属性值为另一个对象，
    #     后面的子树是bestFeat不同取值对应的对象
    myTree = {bestFeatLabel: {}}
    # 1-4 删除标签集合subLabels中对应bestFeat的元素
    del (subLabels[bestFeat])

  # 2 根据特征值划分数据集，构建子树
    # 2-1 得到dataSet中所有的bestFeat的值并组成列表featValues
    featValues = [example[bestFeat] for example in dataSet]
    # 2-2 得出bestFeat的不重复值集合uniqueVals
    uniqueVals = set(featValues)
    # 2-3 遍历uniqueVals，得到每一个bestFeat取值value:
    for value in uniqueVals:
        # 2-3-1 使用splitDataSet函数以dataSet, bestFeat, value为参划分数据集，
        # 得到子数据集subDataSet
        subDataSet = splitDataSet(dataSet, bestFeat, value)
    # 将子数据集subDataSet，subLabels传入createTree函数，返回的值存储为
    # myTree[bestFeatLabel][value]
        myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)

  # 3 返回myTree
    return myTree
