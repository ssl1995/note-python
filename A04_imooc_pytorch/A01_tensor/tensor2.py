import torch

tensor_1 = torch.tensor([[1, 2],
                         [3, 4]])
tensor_2 = tensor_1

# 一、行列
print("第一行:", tensor_1[0])

# 选取 tensor_1 的第一个维度（假设是行）的所有元素；
# 在第二个维度（假设是列）上选取索引为 0 的元素
print("第一列:", tensor_1[:, 0])

# 选取 tensor_1 的所有前面的维度（即除了最后一个维度外的所有维度）；
# 在最后一个维度上选取索引为 -1 的元素。
print("最后一列:", tensor_1[..., -1])

# 二、矩阵运算
# 矩阵乘法
tensor_3 = tensor_1 @ tensor_2
print("矩阵乘法：", tensor_3)

# 元素乘法
tensor_4 = tensor_1 * tensor_2
print("矩阵元素乘法：", tensor_4)

# 元素相加
tensor_5 = tensor_1 + tensor_2
print("矩阵元素相加：", tensor_5)

# 三、拼接
# dim=0，第一个维度拼接
tensor_6 = torch.cat([tensor_1, tensor_2], dim=0)
print("dim=0的拼接：", tensor_6)
tensor_7 = torch.cat([tensor_1, tensor_2], dim=1)
# dim=1，第二个维度拼接
print("dim=1的拼接：", tensor_7)
