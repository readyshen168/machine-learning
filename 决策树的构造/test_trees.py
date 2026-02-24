'''
测试trees.py
'''
import importlib
import pytest
import trees

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="module")
def dataSet():
    """提供trees.create_data_set()返回的数据集和标签，供多个测试函数使用"""
    # 使用importlib.reload确保每次测试都使用模块的最新代码，这在交互式开发时很有用
    trees_module = importlib.reload(trees)
    my_datas, labels = trees_module.createDataSet()
    # 返回数据集，标签集和重新加载的模块
    return my_datas, labels, trees_module


@pytest.fixture(scope="module")
def featSet(dataSet):
    """特征向量集合"""
    return dataSet[0]


class Test_createDataSet:
    """
    测试createDataSet函数
    """

    # 测试1： 验证createDataSet()返回的数据集
    # def test_create_data_set_returns_correct_data(dataSet):

    def test_createDataSet_returns_correctData(self, dataSet):
        """
        测试create_data_set()返回的数据集内容是否正确
        :param dataSet: 使用createDataSet返回的数据，包含my_datas, labels, trees_module
        """
        my_datas, labels, trees_module = dataSet

        # 定义我们期望的数据集结构
        expected_dat = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']
        ]

        # 断言实际返回的数据集与期望的一致
        assert my_datas == expected_dat

    # 测试2： 验证createDataSet()返回的标签集

    def test_createDataSet_returns_correctLabels(self, dataSet):
        """测试createDataSet()返回的标签集是否正确"""
        my_datas, labels, trees_module = dataSet
        expected_labels = ['no surfacing', 'flippers']

        # 断言实际返回的标签集与期望的一致
        assert labels == expected_labels


class Test_calcShannonEnt:
    """ 测试calcShannonEnt函数"""

    # 测试3：验证calcShannonEnt()使用createDataSet()返回的数据计算熵值

    def test_calcShannonEnt_with_createdDataSet(self, dataSet):
        """
        测试calcShannonEnt对标准数据集的熵值计算是否正确

        :param dataSet: 使用createDataSet返回的数据，包含my_datas, labels, trees_module
        """
        my_datas, labels, trees_module = dataSet
        shannon_ent = trees_module.calcShannonEnt(my_datas)

        # 使用pytest.approx处理浮点数比较，避免精度问题
        expected_entropy = 0.9709505944546686
        assert shannon_ent == pytest.approx(expected_entropy)

    # 测试4：验证calcShannonEnt在处理类别完全一致的数据集时熵为0

    def test_calcShannonEnt_with_sameDataSet(self):
        """
        test_calcShannonEnt_with_sameDataSet 
        """
        data_set = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 1, 'yes']
        ]
        trees_module = importlib.reload(trees)
        shannon_ent = trees_module.calcShannonEnt(data_set)

        expected_shannon = 0.0
        assert shannon_ent == expected_shannon

    # 测试5：使用参数化测试多种边界情况

    @pytest.mark.parametrize("test_data, expected_entropy", [
        ([[1, 'a']], 0.0),  # 单一样本，熵值为0
        ([[1, 'a'], [1, 'a']], 0.0),  # 所有样本同类， 熵为0
        ([[1, 'a'], [2, 'b']], 1.0),  # 两个样本各占一半，熵为1
    ])
    def test_calShannonEnt_parameterized(self, test_data, expected_entropy):
        '''
        test_cal_shannon_ent_parameterized 的 Docstring

        :param test_data: 说明
        :type test_data: list[list]
        :param expected_entropy: 说明
        :type expected_entropy: float
        '''
        trees_module = importlib.reload(trees)
        shannon_ent = trees_module.calcShannonEnt(test_data)
        assert shannon_ent == pytest.approx(expected_entropy)


class Test_splitDataSet:
    """测试 splitDataSet函数"""

    def test_split_by_firstFeature_value0(self, featSet):
        """按第一个特征值=0来划分数据集"""
        result = trees.splitDataSet(featSet, axis=0, value=0)
        expected = [
            [1, 'no'],
            [1, 'no'],
        ]
        assert result == expected

    def test_split_by_firstFeature_value1(self, featSet):
        """按第一个特征值=1来划分数据集"""
        result = trees.splitDataSet(featSet, axis=0, value=1)
        expected = [
            [1, 'yes'],
            [1, 'yes'],
            [0, 'no']
        ]
        assert result == expected

    def test_split_by_secondFeature_value0(self, featSet):
        """按第二个特征值 = 0来划分数据集"""

    def test_split_by_secondFeature_value1(self, featSet):
        """按第二个特征值 = 1来划分数据集"""

    def test_split_by_labelFeature(self, featSet):
        """按标签特征划分(边界情况)"""

    def test_noMatchingValue_returns_emptyList(self, featSet):
        """无匹配特征值时返回空列表"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
