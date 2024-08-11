from torch.utils.data import Dataset, DataLoader


# 一、自定义数据集
class MyDataset(Dataset):
    def __init__(self):
        self.x = [i for i in range(10)]
        self.y = [2 * i for i in range(10)]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# 二、初始化数据集
if __name__ == '__main__':
    my_dataset = MyDataset()
    my_dataset_loader = DataLoader(my_dataset,
                                   batch_size=2,  # 每次取值的长度
                                   shuffle=True,  # 是否打乱顺序
                                   num_workers=2  # 启用的线程数量
                                   )

    for x, y in my_dataset_loader:
        print(x, y)
