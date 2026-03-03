'''
放fixture的地方
'''
import pytest
import trees
# import importlib


@pytest.fixture(scope="module")
def sampleDataSet():
    """
    数据样本集，用于大多数测试
    """
    return [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]


@pytest.fixture(scope="module")
def sampleLabels():
    """
    特征标签样本集
    """
    return ['no surfacing', 'flippers']


@pytest.fixture(scope="module")
def dataSet(sampleDataSet, sampleLabels):
    """提供trees.createDataSet()返回的数据集和标签，供多个测试函数使用"""

    my_datas, labels = trees.createDataSet(sampleDataSet, sampleLabels)
    return my_datas, labels


@pytest.fixture(scope="module")
def testDatas(sampleDataSet, sampleLabels):
    """
    返回包含特征和类标签的向量集合
    """
    datas, _ = trees.createDataSet(sampleDataSet, sampleLabels)
    return datas


@pytest.fixture(scope="module")
def testLabels(sampleDataSet, sampleLabels):
    """
    返回类别名称的列表
    """
    _, labels = trees.createDataSet(sampleDataSet, sampleLabels)
    return labels


@pytest.fixture(scope="module")
def featSet(dataSet):
    """返回不带类标签的特征向量集合"""
    # 剔除数据集的最后一列
    feats = [example[:-1] for example in dataSet[0]]
    return feats
