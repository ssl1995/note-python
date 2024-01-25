import numpy as np
import cv2

two_d_matrix = np.random.randint(0, 256, (244, 244), dtype=np.uint8)

# np中是 通道数 × 高 * 宽，转换转换矩阵
three_d_matrix = np.random.randint(0, 256, (3, 244, 244), dtype=np.uint8)
# openCV中，图像的通道顺序为 高 × 宽 × 通道数，所以需要转置。
# (0,1,2) -> (1,2,0)
three_d_matrix_rever = three_d_matrix.transpose(1, 2, 0)

cv2.imshow('two_d_matrix',two_d_matrix)
cv2.imshow('three_d_matrix',three_d_matrix_rever)

cv2.waitKey(0)

# 保存到本地
cv2.imwrite("two_d_matrix.png",two_d_matrix)
cv2.imwrite("three_d_matrix_rever.png",three_d_matrix_rever)

