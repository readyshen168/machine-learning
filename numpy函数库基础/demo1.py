import numpy as np

# 随机数组转换为矩阵
randMat = np.asmatrix(np.random.rand(4, 4))
print(randMat)

# 矩阵逆运算
invRandMat = randMat.I
print(invRandMat)
# 矩阵与其逆矩阵相乘
result = randMat*invRandMat
print(result)

# 单位矩阵
i = np.eye(4)
print(i)

# 相乘结果与单位矩阵的误差
mi = result - i
print(mi)
