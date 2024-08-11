import torch
import torch.nn as nn
import torch.nn.functional as Fun


# 神经网络模版
class Model(nn.Module):
    # 初始化
    def __init__(self):
        # 父类初始化
        super(Model, self).__init__()
        # 卷积层
        # self.conv1 = nn.Conv2d(1, 2, 3)
        # 全连接层
        self.fc_1 = nn.Linear(10, 5, bias=False)

    # 前向传播
    def forward(self, x):
        # 激活函数：relu
        # x = Fun.relu(self.conv1(x))

        x = self.fc_1(x)
        return x


if __name__ == '__main__':
    model = Model()
    print(model)

    input = torch.randn(10)

    output = model(input)

    print(output)
