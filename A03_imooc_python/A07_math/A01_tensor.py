import numpy as np

# 一维张量
vector= np.array([1, 2, 3])

# 二维张量
matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [1, 2, 3]])
# 三维张量
tensor= np.array([
    [[1, 2], [3, 4]],
    [[1, 2], [3, 4]],
    [[1, 2], [3, 4]]
])

# 范数 = 可以理解为张量的长度
tensor_norm= np.linalg.norm(tensor)

print(f"tensor_norm：{ten sor_norm}")
