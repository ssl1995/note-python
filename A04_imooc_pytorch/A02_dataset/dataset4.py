import torchvision
from torch.utils.data import DataLoader

transform = torchvision.transforms.ToTensor()

# 下载官方数据集
# MNIST训练集
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

# MNIST测试集
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

for x, y in train_loader:
    print(x.shape, y)

for x, y in test_loader:
    print(x.shape, y)
