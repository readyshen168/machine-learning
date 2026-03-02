'''
测试trees.calShannonSimple方法
'''
import math
from math import log
import pytest
from trees import calShannonSimple

diffValue = 1e-9


def test_valid_input():
    '''
    测试有效的概率列表
    '''

    # 简单的二分类
    entropy = calShannonSimple([0.5, 0.5])
    # print(f'[0.5, 0.5]: {entropy}')
    assert math.isclose(entropy, 1.0, rel_tol=diffValue)

    # 三分类均匀分布
    data = [1/3, 1/3, 1/3]
    entropy = calShannonSimple(data)
    # print(f'[1/3, 1/3, 1/3]: {entropy}')
    expected = -3 * (1/3) * log(1/3, 2)
    assert math.isclose(entropy, expected, rel_tol=diffValue)


def test_floatingPoint_precision():
    '''
    测试浮点数精度问题
    '''

    '''数据集总和大于1'''
    # 在允许误差范围内
    data = [0.3333333333333333, 0.3333333333333333, 0.3333333333333334]
    result = calShannonSimple(data)
    assert result >= 0

    # 在允许误差范围外: 0.001
    data = [0.34, 0.33, 0.331]

    # 使用pytest.raises捕获异常
    with pytest.raises(ValueError) as exc_info:
        calShannonSimple(data)

    # 验证异常信息包含特定字符串
    assert "概率和必须为1" in str(exc_info.value)
    assert "1.001" in str(exc_info.value)

    '''数据集总和小于1'''
    # 在允许误差范围内: 0.9999999999
    data = [0.3333333333, 0.3333333333, 0.3333333333]
    result = calShannonSimple(data)
    assert result >= 0

    # 在允许误差范围外: 0.99999999
    data = [0.33333333, 0.33333333, 0.33333333]
    pattern = r"概率和必须为1， 但得到0\.[0-9]*"
    with pytest.raises(ValueError, match=pattern) as excInfo:
        calShannonSimple(data)

    exception = excInfo.value
    print(f"捕获的异常: {exception}")
    assert isinstance(exception, ValueError)


if __name__ == "__main__":
    # 可以单独运行测试
    pytest.main([__file__, "-v"])
