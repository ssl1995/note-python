import torch
import matplotlib.pyplot as plt
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

'''
训练函数的模版:
数据集、数据加载器、模型、损失函数、优化器、数据保存
'''


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


# 定义训练函数-优化
def model_train(dataLoader, model, loss_fn, optimizer):
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
def model_test(dataLoader, model, loss_fn):
    data_total_size = len(dataLoader.dataset)
    data_batch_size = len(dataLoader)

    model.eval()

    test_loss = 0
    correct = 0

    # 测试集，不需要进行梯度计算 = 提高计算效率并减少内存使用
    with torch.no_grad():
        for x, y in tqdm(dataLoader):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_fn(pred, y)

            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= data_batch_size
    correct /= data_total_size

    return round(correct, 3), round(test_loss, 3)


# 保存数据
def writeData(txt_log_name, tensorboard_writer, epoch, train_acc, train_loss, test_acc, test_loss):
    # 1、保存到本地文档
    with open(txt_log_name, "a+") as f:
        f.write(f"{epoch}\t{train_acc}\t{train_loss}\t{test_acc}\t{test_loss}\n")

    # 2、保存到tensorboard
    for name, param in model.named_parameters():
        if 'linear' in name:
            # add_histogram=模型参数可视化
            tensorboard_writer.add_histogram(name, param.clone().cuda().data.numpy(), global_step=epoch)

    tensorboard_writer.add_scalar('Accuracy/train', train_acc, epoch)
    tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
    tensorboard_writer.add_scalar('Accuracy/test', test_acc, epoch)
    tensorboard_writer.add_scalar('Loss/test', test_loss, epoch)


# 画图函数
def plot_txt(log_txt_loc):
    with open(log_txt_loc, "r") as f:
        log_data = f.read()

    # 解析日志数据
    epoch_s = []
    train_acc_s = []
    train_loss_s = []
    test_acc_s = []
    test_loss_s = []

    for line in log_data.strip().split("\n"):
        epoch, train_acc, train_loss, test_acc, test_loss = line.split("\t")

        epoch_s.append(int(epoch))
        train_acc_s.append(float(train_acc))
        train_loss_s.append(float(train_loss))
        test_acc_s.append(float(test_acc))
        test_loss_s.append(float(test_loss))

    # 构建折线图
    plt.figure(figsize=(10, 5))

    # 训练数据
    plt.subplot(1, 2, 1)
    plt.plot(epoch_s, train_acc_s, label='Train Accuracy')
    plt.plot(epoch_s, test_acc_s, label='Test Accuracy')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    # 设置横坐标刻度为整数
    plt.xticks(range(min(epoch_s), max(epoch_s) + 1))

    # 测试数据
    plt.subplot(1, 2, 2)
    plt.plot(epoch_s, train_loss_s, label='Train Loss')
    plt.plot(epoch_s, test_loss_s, label='Test Loss')
    plt.title('Testing Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    # 设置横坐标刻度为整数
    plt.xticks(range(min(epoch_s), max(epoch_s) + 1))

    plt.tight_layout()
    plt.show()


# 主函数
if __name__ == '__main__':
    log_root = "logs"
    log_txt_loc = os.path.join(log_root, "log.txt")

    # 指定TensorBoard数据的报错地址
    tensorboard_writer = SummaryWriter(log_root)

    if os.path.isdir(log_root):
        pass
    else:
        os.mkdir(log_root)

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
    model = NeuralNetWork().to(device)
    print(model)

    # 模拟输入，大小和输入相同即可
    init_img = torch.zeros((1, 1, 28, 28), device=device)
    tensorboard_writer.add_graph(model, init_img)

    # 损失函数，CrossEntropyLoss=交叉熵损失函数，常用语分类任务
    # 如果 loss_fn 是交叉熵损失（例如 torch.nn.CrossEntropyLoss），它通常会产生较大的数值，尤其是在模型刚开始训练时，分类错误较多的情况下。
    # 如果 loss_fn 是均方误差损失（例如 torch.nn.MSELoss），并且输出和目标值的差距较大时，损失值也会很大。
    loss_fn = nn.CrossEntropyLoss()

    # 优化器，用于更新模型参数，lr=1e-3是学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    best_acc = 0
    epochs = 5
    for t in range(epochs):
        print(f"\n Epoch {t + 1}\n-------------------------------")
        # 训练集结果
        train_acc, train_loss = model_train(train_dataLoader, model, loss_fn, optimizer)
        test_acc, test_loss = model_test(test_dataLoader, model, loss_fn)
        writeData(log_txt_loc, tensorboard_writer, t, train_acc, train_loss, test_acc, test_loss)

        # 保存最佳结果
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(log_root, "best.pth"))

        # 保存最新结果
        torch.save(model.state_dict(), os.path.join(log_root, "last.pth"))

    print("Done!")

    plot_txt(log_txt_loc)
    # 关闭tensorboard
    tensorboard_writer.close()
