"""测试majorityCnt方法"""
import pytest
import trees


class TestMajorityCnt:
    """测试majorityCnt方法"""

    def test_majority_cnt_returns_most_common(self):
        """测试返回最常见的类别"""
        class_list = ['yes', 'yes', 'no', 'yes', 'no']
        result = trees.majorityCnt(class_list)
        assert result == 'yes'

    def test_majority_cnt_tie_breaking(self):
        """测试平局时的处理"""
        class_list = ['yes', 'no', 'yes', 'no']
        result = trees.majorityCnt(class_list)
        # 取决于实现，可能是第一个遇到的最大值
        assert result in ['yes', 'no']

    def test_majority_cnt_single_class(self):
        """测试单一类别"""
        class_list = ['yes', 'yes', 'yes']
        result = trees.majorityCnt(class_list)
        assert result == 'yes'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
