'''
放fixture的地方
'''
import pytest
import trees
# import importlib


@pytest.fixture(scope="module")
def dataSet():
    """提供trees.create_data_set()返回的数据集和标签，供多个测试函数使用"""
    # 使用importlib.reload确保每次测试都使用模块的最新代码，这在交互式开发时很有用
    # trees_module = importlib.reload(trees)
    # my_datas, labels = trees_module.createDataSet()
    my_datas, labels = trees.createDataSet()
    # 返回数据集，标签集和重新加载的模块
    return my_datas, labels


@pytest.fixture(scope="module")
def testDatas():
    """
    返回包含特征和类标签的向量集合
    """
    datas, _ = trees.createDataSet()
    return datas


@pytest.fixture(scope="module")
def testLabels():
    """
    返回类别名称的列表
    """
    _, labels = trees.createDataSet()
    return labels


@pytest.fixture(scope="module")
def featSet(dataSet):
    """返回不带类标签的特征向量集合"""
    # 剔除数据集的最后一列
    feats = [example[:-1] for example in dataSet[0]]
    return feats
