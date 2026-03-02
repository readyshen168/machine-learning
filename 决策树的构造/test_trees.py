'''
测试trees.py
'''
import pytest

# pylint: disable=redefined-outer-name


class Test_createDataSet:
    """
    测试createDataSet函数
    """

    # 测试1： 验证createDataSet()返回的数据集
    # def test_create_data_set_returns_correct_data(dataSet):
    def test_createDataSet_returns_correctData(self, testDatas):
        """
        测试create_data_set()返回的数据集内容是否正确
        :param dataSet: 使用createDataSet返回的数据，包含my_datas, labels, trees_module
        """
        # 定义我们期望的数据集结构
        expected_dat = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']
        ]

        # 断言实际返回的数据集与期望的一致
        assert testDatas == expected_dat

    # 测试2： 验证createDataSet()返回的标签集

    def test_createDataSet_returns_correctLabels(self, testLabels):
        """测试createDataSet()返回的标签集是否正确"""

        # my_datas, labels, trees_module = dataSet
        expected_labels = ['no surfacing', 'flippers']

        # 断言实际返回的标签集与期望的一致
        assert testLabels == expected_labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
