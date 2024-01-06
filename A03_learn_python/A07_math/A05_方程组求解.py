import numpy as np
import cv2

A = np.array([[1, 2, 3],
              [1, 6, 7],
              [1, 10, 6]])

b = np.array([5, 9, 8])

# 求解方程组
#  linalg 模块包含线性代数中的函数方法，用于求解矩阵的逆矩阵、求特征值、解线性方程组以及求行列式等
x = np.linalg.solve(A, b)

print(f"矩阵求解:\n{x}")
