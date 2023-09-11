import torch

# Tensor的连接操作
## cat：拼接，原维度不变

## stack:dim方向升级维度拼接
A = torch.arange(0, 4)
B = torch.arange(5, 9)
C = torch.stack((A, B), 0)
print(C)
D = torch.stack((A, B), 1)
print(D)

# Tensor的切分操作
## chunk:按照声明的dim，尽可能平均的切分
A = torch.tensor([1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10])
# 将A尽可能的平均切成2块
B = torch.chunk(A, 2, 0)
# (tensor([1, 2, 3, 4, 4, 5]), tensor([ 6,  7,  8,  9, 10]))
print(B)
# chunk如果不能平均分，17/4=4.25，向上取整=5，再生成若干个长度5，最后不够的放在一块
A = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
B = torch.chunk(A, 4, 0)

# 二维的分块，按照dim的方向切分
A = torch.ones(4, 4)
B = torch.chunk(A, 2, 0)

# split：每份按照确定的大小切分
# 第二个参数：整数，每块大小是确定的，不足的先填上，剩余的当做一块
A = torch.rand(4, 4)
print(A)
B = torch.split(A, 2, 0)
print(B)

# 第二个参数：不是整数，就尽可能的凑够每一个结果，剩下的单独放一块
C = torch.split(A, 3, 0)
print(C)

# 第三个参数：沿着第 0 维进行切分，每一个结果对应维度上的尺寸或者说大小，分别是 2（行），3（行）。
A = torch.rand(5, 4)
print(A)
B = torch.split(A, (2, 3), 0)
print(B)

# unbind:获取每个channel的数据 = 降维切分的方式
A = torch.arange(0, 16).view(4, 4)
print(A)
B = torch.unbind(A, 0)
print(B)

# Tensor的索引操作
## idenx_select: 想要局部的哪块部分
A = torch.arange(0, 16).view(4, 4)
print(A)
B = torch.index_select(A, 0, torch.tensor([1, 3]))
print(B)

## masked_selck: A 中“满足 B 里面元素值为 True 的”对应位置的数据。
A = torch.rand(5)
print(A)
C = torch.masked_select(A, A > 0.3)
print(C)

# 练习-获取第一行的第一个、第二行的第一和二个、第三行的第一个
A = torch.tensor([[4, 5, 7], [3, 9, 8], [2, 3, 4]])
B = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 0]])
# tensor([4, 3, 9, 2])
C = torch.masked_select(A, A * B != 0)
print(C)
