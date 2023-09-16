

import torch
import torchvision.models as models

# 模型微调
# 加载预训练模型
googlenet = models.googlenet(pretrained=True)

# 提取分类层的输入参数
fc_in_features = googlenet.fc.in_features
print(fc_in_features)  # 1024

# 提出分类层的输出参数
fc_out_features = googlenet.fc.out_features
print(fc_out_features)  # 1000

# 修改预训练模型的输出分类数
googlenet.fc = torch.nn.Linear(fc_in_features, 10)


import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# make_grid拼接成网格
# 加载MNIST数据集
mnist_dataset = datasets.MNIST(
    root='./root', train=False, transform=transforms.ToTensor(),
    target_transform=None,
    download=True
)

# 取32张图片的tensor
tersor_dataloader = DataLoader(dataset=mnist_dataset, batch_size=32)

data_iter = iter(tersor_dataloader)
img_tensor, labeL_tensor = data_iter._next_data()
print(img_tensor.shape)

# 将23张图片拼接在一个网络中
grid_tensor = torchvision.utils.make_grid(img_tensor, nrow=8, padding=2)
grid_img = transforms.ToPILImage()(grid_tensor)

display(grid_img)

## save_image保存
# 输入为一张图片的tensor 直接保存
torchvision.utils.save_image(grid_tensor, 'grid.jpg')
# 输入为List，调佣grid_img函数后保存
torchvision.utils.save_image(img_tensor, 'grid2.jpg', nrow=5, padding=2)
