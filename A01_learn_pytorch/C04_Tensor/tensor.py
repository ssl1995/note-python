import torch

# 获取形状
a = torch.zeros(2, 3, 5)
print('---------')
# torch.Size([2, 3, 5])
print(a.shape)
print('---------')
# torch.Size([2, 3, 5])
print(a.size())
# 数据量大小
print(a.numel)
print("----------")

# 矩阵转秩
## 缺点：以下2个方法会让数据变的内存不连续
x = torch.rand(2, 3, 5)
print(x.shape)  # torch.Size([2, 3, 5])
x = x.permute(2, 1, 0)
print(x.shape)  # torch.Size([5, 3, 2])
x = x.transpose(0, 1)  # 交换0和1位置
print(x.shape)  # torch.Size([3, 5, 2])
print("----------")

# 形状变换
x = torch.rand(4, 4)
print(x.shape)
## view需要内存连续才行
x = x.view(2, 8)
print(x.shape)
x = x.reshape(2, 8)
print(x.shape)
print("----------")

# 增减维度
## squeeze:只能往指定维度=1的数据进行减增
x = torch.rand(2, 1, 3)
y = x.squeeze(2)
print(y.shape)

z = x.unsqueeze(2)
# torch.Size([2, 1, 1, 3])
print(z.shape)


