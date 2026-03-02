"""
测试trees模块中的splitDataSet方法
"""

from trees import splitDataSet


class Test_splitDataSet:
    """测试 splitDataSet函数"""

    def test_split_by_firstFeature_value0(self, testDatas):
        """按第一个特征值=0来划分数据集"""
        result = splitDataSet(testDatas, axis=0, value=0)
        expected = [
            [1, 'no'],
            [1, 'no'],
        ]
        assert result == expected

    def test_split_by_firstFeature_value1(self, testDatas):
        """按第一个特征值=1来划分数据集"""
        result = splitDataSet(testDatas, axis=0, value=1)
        expected = [
            [1, 'yes'],
            [1, 'yes'],
            [0, 'no']
        ]
        assert result == expected

    def test_split_by_secondFeature_value0(self, testDatas):
        """按第二个特征值 = 0来划分数据集"""

    def test_split_by_secondFeature_value1(self, testDatas):
        """按第二个特征值 = 1来划分数据集"""

    def test_split_by_labelFeature(self, testDatas):
        """按标签特征划分(边界情况)"""

    def test_noMatchingValue_returns_emptyList(self, testDatas):
        """无匹配特征值时返回空列表"""
