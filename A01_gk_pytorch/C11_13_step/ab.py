import LeNet  # 假定我们使用的模型叫做LeNet，首先导入模型的定义类
import torch.optim as optim  # 引入PyTorch自带的可选优化函数

## 这是一个抽象框架，不能运行，用于理清深度学习流程
## 11到13节，学习了损失函数、反向传播和优化方法（梯度下降）
# 1.模型定义
# 2.损失函数定义
# 3.优化器定义

# 1.模型定义
net = LeNet()  # 声明一个LeNet的实例

# 2.损失函数
criterion = nn.CrossEntropyLoss()  # 声明模型的损失函数，使用的是交叉熵损失函数

# 3.优化函数：使用SDG
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 声明优化函数，我们使用的就是之前提到的SGD，优化的参数就是LeNet内部的参数，lr即为之前提到的学习率

# 下面开始训练
# 这里是不能跑的，注意下
for epoch in range(30):  # 设置要在全部数据上训练的次数

    for i, data in enumerate(traindata):
        # data就是我们获取的一个batch size大小的数据

        inputs, labels = data  # 分别得到输入的数据及其对应的类别结果
        # 首先要通过zero_grad()函数把梯度清零，不然PyTorch每次计算梯度会累加，不清零的话第二次算的梯度等于第一次加第二次的
        optimizer.zero_grad()
        # 获得模型的输出结果，也即是当前模型学到的效果
        outputs = net(inputs)
        # 获得输出结果和数据真正类别的损失函数
        loss = criterion(outputs, labels)
        # 算完loss之后进行反向梯度传播，这个过程之后梯度会记录在变量中
        loss.backward()
        # 用计算的梯度去做优化
        optimizer.step()
