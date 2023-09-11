import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


# 生成数据
data_tensor = torch.randn(10, 3)
# 标签时0或1
target_tensor = torch.randint(2, (10,))
# tensor([0, 1, 0, 0, 0, 1, 0, 0, 1, 0])
print(target_tensor)
# 查看数据集大小
my_dataset = MyDataset(data_tensor, target_tensor)
# 使用索引调用数据
# Dataset size: 10
print('Dataset size:', len(my_dataset))
# tensor_data[0]: (tensor([ 1.1965, -1.3266, -2.6503]), tensor(0))
print('tensor_data[0]:', my_dataset[0])



from torch.utils.data import DataLoader

tensor_dataloader = DataLoader(dataset=my_dataset,  # 传入的数据集，必须参数
                               batch_size=2,  # 输出的batch大小
                               shuffle=True,  # 数据是否打乱
                               num_workers=0)  # 进程数，0=只有主进程
# 以循环形式输出
for data, target in tensor_dataloader:
    print(data, target)

print('One bath tensor data: ', iter(tensor_dataloader).next())
