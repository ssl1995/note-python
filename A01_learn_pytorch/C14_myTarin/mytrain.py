import numpy as np
import random
import mytorch
from mytorch import nn

from matplotlib import pyplot as plt

w = 2
b = 3
xlim = [-10, 10]
x_train = np.random.randint(low=xlim[0], high=xlim[1], size=30)

y_train = [w * x + b + random.randint(0, 2) for x in x_train]


# 定义模型
class LinerModel(nn.Module):

    # 必须重写init
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    # 正向传播函数
    def forward(self, input):
        return (input * self.weight) + self.bias


# 训练模型
model = LinerModel()
# 优化方法：SGD随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

y_train = torch.tensor(y_train, dtype=torch.float32)

for _ in range(1000):
    input = torch.from_numpy(x_train)
    output = model(input)
    loss = nn.MSELoss()(output, y_train) # 损失函数 = MSE均方误差
    model.zero_grad()
    loss.backward()
    optimizer.step()

for parameter in model.named_parameters():
    print(parameter)
# ('weight', Parameter containing:
# tensor([1.9728], requires_grad=True))
# ('bias', Parameter containing:
# tensor([3.4435], requires_grad=True))

# 模型的保存和加载
# 保存：已经训练好的参数
torch.save(model.state_dict(),'./linear_model.pth')
# 使用曾经保存过的参数
liner_model = LinerModel()
liner_model.load_state_dict(torch.load('./linear_model.pth'))
liner_model.eval() # 模型评估状态
for parameter in liner_model.named_parameters():
    print(parameter)

# 保存网络结构和参数
torch.save(model,'./linear_model_with_arc.pth')
torch_model_2 = torch.load('./linear_model_with_arc.pth')
torch_model_2.eval()
for parameter in torch_model_2.named_parameters():
    print(parameter)

