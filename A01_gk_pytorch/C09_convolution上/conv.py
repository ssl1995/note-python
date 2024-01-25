import torch
import torch.nn as nn

# Conv2d是touch.nn中提供的卷积方法，点击源码进去看参数说明
# class torch.nn.Conv2d(
# in_channels,  # 输入特征图的通道数
# out_channels,  # 输出特征图的通道数
# kernel_size,  # 卷积核的大小
# stride=1,  # 步长
# padding=0,  # 补零的方式
# dilation=1,
# groups=1,
# bias=True,  # 是否使用偏移量
# padding_mode='zeros',
# device=None,
# dtype=None)

input_feat = torch.tensor([[4, 1, 7, 5], [4, 4, 2, 5], [7, 7, 2, 4], [1, 0, 2, 4]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# 创建一个2*2的卷积
# conv2d = nn.Conv2d(1, 1, (2, 2), stride=1, padding='same', bias=True)
# 默认情况下是随机初始化的。一般情况下，我们不会人工强行干预卷积核的初始化，

# 人为干预下卷积核
conv2d = nn.Conv2d(1, 1, (2, 2), stride=1, padding='same', bias=False)
# 卷积核要有四个维度(输入通道数，输出通道数，高，宽)
kernels = torch.tensor([[[[1, 0], [2, 1]]]], dtype=torch.float32)
conv2d.weight = nn.Parameter(kernels, requires_grad=False)
print(conv2d.weight)
print(conv2d.bias)

output = conv2d(input_feat)
print(output)
