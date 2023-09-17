import torch
import torch.nn as nn

# DW卷积代码
# 生成一个三通道的5x5特征图
x = torch.rand((3, 5, 5)).unsqueeze(0)
print(x.shape)
# 输出：
torch.Size([1, 3, 5, 5])
# 请注意DW中，输入特征通道数与输出通道数是一样的
in_channels_dw = x.shape[1]
out_channels_dw = x.shape[1]
# 一般来讲DW卷积的kernel size为3
kernel_size = 3
stride = 1
# DW卷积groups参数与输入通道数一样
dw = nn.Conv2d(
    in_channels_dw,
    out_channels_dw,
    kernel_size,
    stride,
    groups=in_channels_dw  # group参数控制输入、输出特征图的分组情况
    # groups=1, =标准卷积
    # groups≠1,会将输入特征图分成groups个分组，每个组都有自己对应的卷积核，然后分组卷积；不为1时，必须能整除in_channels和out_channels
)

# PW卷积代码
in_channels_pw = out_channels_dw
out_channels_pw = 4
kernel_size_pw = 1
pw = nn.Conv2d(
    in_channels_pw,
    out_channels_pw,
    kernel_size_pw,  # PW卷积核参数必=1
    stride)
out = pw(dw(x))
print(out.shape)
