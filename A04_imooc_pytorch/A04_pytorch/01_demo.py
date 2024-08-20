import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

'''
训练函数的模版:
数据集、数据加载器、模型、损失函数、优化器、数据保存
'''

# 数据集
train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

# 数据加载器
batch_size = 64
train_dataLoader = DataLoader(train_data, batch_size=batch_size)
test_dataLoader = DataLoader(test_data, batch_size=batch_size)

# 打印数据的格式，便于书写模型
print("train_dataLoader-come")

for x, y in train_dataLoader:
    # Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


print("test_dataLoader-come")
for x, y in test_dataLoader:
    # Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetWork(nn.Module):
    def __init__(self):
        super().__init__()
        # 将输入张量展平，便于输入全连接层和线性层
        self.flatten = nn.Flatten()
        self.liner_relu_stack = nn.Sequential(
            # 28*28是print(f"Shape...)拿到的数据集尺寸大小
            # 512是自己定义的模型输出特征数
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.liner_relu_stack(x)
        return x


model = NeuralNetWork().to(device)
print(model)

# 损失函数，CrossEntropyLoss=交叉熵损失函数，常用语分类任务
loss_fn = nn.CrossEntropyLoss()

# 优化器，用于更新模型参数，lr=1e-3是学习率
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# 定义训练函数
def train_fun(dataLoader, model, loss_fn, optimizer):
    size = len(dataLoader.dataset)
    # 模型调到训练模式
    model.train()

    for batch, (x, y) in enumerate(dataLoader):
        x, y = x.to(device), y.to(device)

        # 将输入的x进行模型训练
        pred = model(x)
        # 计算预测结果和输出结果相差多少
        loss = loss_fn(pred, y)
        # 反向传播和优化
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        optimizer.zero_grad()  # 清空梯度

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            # loss:{loss:>7f} 表示损失值 loss 会被格式化为浮点数，保留 7 位宽度，不足的部分会在左边填充空格。
            # :f 表示浮点数格式，> 表示右对齐，7 表示总共占用 7 位字符宽度。
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")


# 定义测试函数
def test_fun(dataLoader, model, loss_fn):
    size = len(dataLoader.dataset)
    num_batches = len(dataLoader)
    # 将模型切换到验证模型
    # 说明模型只会进行推理验证，不会进行反向传播和优化参数
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad(): # 不需要计算梯度
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    print(f"Test Result:\n Accuracy:{(100 * correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")



if __name__ == '__main__':
    # 定义循环次数，每次循环里面，先训练，再测试
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_fun(train_dataLoader, model, loss_fn, optimizer)
        test_fun(test_dataLoader, model, loss_fn)
    print("Done!")

    # 保存
    torch.save(model.state_dict(),'model.pth')
    print("保存成功")