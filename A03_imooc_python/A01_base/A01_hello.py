import numpy as np

# 创建一个二维数组，形状为 (1, 4)
Y = np.array([[0, 1, 1, 0]])
print("Y 的形状:", Y.shape)  # 输出: (1, 4)

# 创建另一个二维数组，形状为 (4, 3)
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print("A 的形状:", A.shape)  # 输出: (4, 3)

# 现在我们可以直接进行矩阵乘法
result = np.dot(Y, A)  # 注意，我们将 Y 转置以匹配 A 的列数
print("结果的形状:", result.shape)  # 输出: (4, 1)
