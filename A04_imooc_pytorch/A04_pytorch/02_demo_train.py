import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

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
# 如果 loss_fn 是交叉熵损失（例如 torch.nn.CrossEntropyLoss），它通常会产生较大的数值，尤其是在模型刚开始训练时，分类错误较多的情况下。
# 如果 loss_fn 是均方误差损失（例如 torch.nn.MSELoss），并且输出和目标值的差距较大时，损失值也会很大。
loss_fn = nn.CrossEntropyLoss()

# 优化器，用于更新模型参数，lr=1e-3是学习率
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# 定义训练函数-优化
def train_fun(dataLoader, model, loss_fn, optimizer):
    data_total_size = len(dataLoader.dataset)
    data_batch = len(dataLoader)

    model.train()

    # 总的损失
    loss_total = 0
    # 预测正确的个数
    correct = 0
    for x, y in tqdm(dataLoader):
        x, y = x.to(device), y.to(device)

        '''
        预测结果:tensor([[ 2.0645e-02, -1.0214e-01,  7.6782e-02, -1.2014e-02,  1.3320e-01,
         -1.8308e-01,  1.6507e-02, -1.1457e-01,  2.1594e-01,  1.5671e-02],
        [-3.9951e-03, -2.3991e-02,  2.6100e-02,  1.5857e-02,  3.9585e-02,
        '''
        pred = model(x)

        '''
        loss函数结果:2.235142230987549
        '''
        loss = loss_fn(pred, y)
        loss_total += loss.item()

        '''
        pred.argmax(1) == y的结果:tensor([ True, False, False, False, False, False,  True, False,  True, False,
         True,  True, False, False,  True, False,  True, False, False, False,
        False, False, False, False, False, False, False, False,  True, False,
        '''
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss_avg = loss_total / data_batch
    correct = correct / data_total_size

    return round(correct, 3), round(loss_avg, 3)


# 定义测试函数
def test_fun(dataLoader, model, loss_fn):
    size = len(dataLoader.dataset)
    num_batches = len(dataLoader)
    # 将模型切换到验证模型
    # 说明模型只会进行推理验证，不会进行反向传播和优化参数
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():  # 不需要计算梯度
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    print(f"Test Result:\n Accuracy:{(100 * correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")

train_acc_list = []
train_loss_list = []

if __name__ == '__main__':
    # 定义循环次数，每次循环里面，先训练，再测试
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_acc,train_loss  = train_fun(train_dataLoader, model, loss_fn, optimizer)
        train_acc_list.append(train_acc_list)
        train_loss_list.append(train_loss)

        test_fun(test_dataLoader, model, loss_fn)
    print("Done!")

    # 保存
    torch.save(model.state_dict(), 'model.pth')
    print("保存成功")


import matplotlib.pyplot as plt

x_list = [i+1 for i in range(len(train_acc_list))]
plt.plot(x_list, train_acc_list, label="Train")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(x_list, train_loss_list, label="Train")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()