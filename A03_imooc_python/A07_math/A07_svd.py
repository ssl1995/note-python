import cv2

import numpy as np

image = cv2.imread('robot.png', 0)

# 对图像进行svg分解
U, S, V = np.linalg.svd(image.astype(np.float64), full_matrices=False)

# 定义要保留的奇异值数量
k = 10
# diag:matlab中用来对矩阵的对角元素进行提取和创建对角阵的函数。
s_k = np.diag(S[:k])

# 重构图像
# dot:获取两个元素a,b的乘积
compressed_image = np.dot(U[:, :k], np.dot(s_k, V[:k, :]))

compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

cv2.imshow("image", image)
cv2.imshow("compressed_image", compressed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
