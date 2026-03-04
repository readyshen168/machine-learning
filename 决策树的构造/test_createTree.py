"""测试createTree方法"""

import pytest
import trees


class Test_createTree:
    """测试createTree"""

    def test_createTree_returns_expected_structures(self, dataSet):
        """处理数据集，得出预想的决策树结构"""
        datas, labels = dataSet

        resultTree = trees.createTree(datas, labels)

        expectedTree = {
            'no surfacing': {
                0: 'no',
                1: {
                    'flippers': {
                        0: 'no',
                        1: 'yes'
                    }
                }
            }
        }

        assert resultTree == expectedTree

    def test_createTree_with_pure_label(self):
        """测试纯类别数据集（所有样本同一类别）"""

        datas = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'yes'],
        ]

        labels = ['f1', 'f2']

        resultTree = trees.createTree(datas, labels)

        expectedTree = 'yes'

        assert resultTree == expectedTree

    def test_createTree_with_single_feature(self):
        """测试只有一个特征的数据集"""
        datas = [
            [1, 'yes'],
            [1, 'yes'],
            [0, 'no'],
        ]

        labels = ['f1']

        result_tree = trees.createTree(datas, labels)

        expected_tree = {
            'f1': {
                0: 'no',
                1: 'yes'
            }
        }

        assert result_tree == expected_tree


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
