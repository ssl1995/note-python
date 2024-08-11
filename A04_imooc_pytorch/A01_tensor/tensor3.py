import torch
import numpy as np

# 1、np转torch
n = np.ones(5)  # 列表中的 1. 表示浮点数 1.0。
t = torch.from_numpy(n)
print("t:", t)

# 2、torch转np
n_2 = t.numpy()
print("n_2:", n_2)

