import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict


# 定义模块
class MyBlock(nn.Module):
    #
    # in_channels:输入通道的大小
    # out_channels:输出通道的大小
    #
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 填充：padding，用于x+y的大小不一致，需要填充
        self.c1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.reLu = nn.ReLU()
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = x
        x = self.c1(x)
        x = self.bn1(x)
        x = self.reLu(x)
        x = self.c2(x)
        # 问题：如果一个x+y，大小不一致，会报错，所以需要填充padding
        res = x + y
        return res


class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积
        self.c1 = nn.Conv2d(3, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        # block层
        self.body = nn.Sequential(
            OrderedDict([
                ('block_1', MyBlock(64, 64)),
                ('block_2', MyBlock(64, 64)),
                ('block_3', MyBlock(64, 64)),
                ('block_4', MyBlock(64, 64))
            ])
        )
        # 全连接层
        self.tail = nn.Sequential(
            # 64 * 220 * 220的值需要用summary的Debug出来确定
            nn.Linear(64 * 220 * 220, 512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.body(x)
        # 在导入线性层时，只需要2个维度，需要拉平所有的参数
        x = x.view(-1, 64 * 220 * 220)
        x = self.tail(x)

        return x


if __name__ == '__main__':
    # 输入放到GPU上
    input = torch.rand((5, 3, 224, 224)).cuda()

    myModel = myNet()
    # 输出模型也放到GPU上
    myModel = myModel.cuda()

    output = myModel(input)
    print(output.shape)

    summary(myModel, (3, 224, 224))
