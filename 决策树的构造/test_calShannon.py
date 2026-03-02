'''
测试trees.calShannonSimple方法
'''
import pytest
from trees import calcShannonEnt


class Test_calcShannonEnt:
    """ 测试calcShannonEnt函数"""

    # 测试1：验证calcShannonEnt()使用createDataSet()返回的数据计算熵值

    def test_calcShannonEnt_with_createdDataSet(self, testDatas):
        """
        测试calcShannonEnt对标准数据集的熵值计算是否正确

        :param featSet: 来自conftest.py中fixture返回的数据集
        """
        shannon_ent = calcShannonEnt(testDatas)

        # 使用pytest.approx处理浮点数比较，避免精度问题
        expected_entropy = 0.9709505944546686
        assert shannon_ent == pytest.approx(expected_entropy)

    # 测试2：验证calcShannonEnt在处理类别完全一致的数据集时熵为0

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
        shannon_ent = calcShannonEnt(data_set)

        expected_shannon = 0.0
        assert shannon_ent == expected_shannon

    # 测试3：使用参数化测试多种边界情况

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
        shannon_ent = calcShannonEnt(test_data)
        assert shannon_ent == pytest.approx(expected_entropy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
