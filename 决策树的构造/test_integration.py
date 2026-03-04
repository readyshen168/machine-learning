"""
决策树算法集成测试
测试从数据到决策树的完整流程
"""
import pytest
import math
import trees


class TestDecisionTree_with_noInfoGain_Integration:
    """当数据集无信息增益时的决策树集成测试类"""

    def setup_method(self):
        """测试前的准备工作"""
        # 定义测试数据集，无信息增益
        self.noInfoGain_dataset = [
            [1, 'yes'],
            [1, 'no'],
            [0, 'yes'],
            [0, 'no'],
        ]

        self.noInfoGain_labels = ['feature1']

    def test_calcShannonEnt_no_infoGain(self):
        """测试无信息增益数据集的熵计算"""
        # 计算self.noInfoGain_dataset数据集的熵值entropy
        # 期望熵值为1.0
        # assert语句：math.isclose()方法比较两个熵值，并打印文字：
        # 期望熵值为：xxx, 实际熵值为：xxx

    def test_chooseBestFeatureToSplit_no_infoGain(self):
        """测试无信息增益时的最佳特征选择"""
        # 计算无信息增益数据集的best_feature, 该值的期待值为-1
        # assert输出： 当没有信息增益时，期望返回-1， 实际得到：

    def test_createTree_handles_no_infoGain(self):
        """测试createTree处理无信息增益时的情况"""
        # 这里我们会遇到问题，因为chooseBestFeatureToSplit返回-1
        # 而createTree会尝试访问labels[-1]，导致IndexError

        # 修改createTree以处理这种情况
        # 在实际实现中，我们需要修改createTree函数
        # 在调用chooseBestFeatureToSplit后检查返回值

        # 先捕获异常，验证当前实现确实会失败
        with pytest.raises(IndexError) as excInfo:
            tree = trees.createTree(
                self.noInfoGain_dataset, self.noInfoGain_labels)


class TestDecisionTree_with_standardDataset_Integration:
    """标准数据集的决策树集成测试类"""
