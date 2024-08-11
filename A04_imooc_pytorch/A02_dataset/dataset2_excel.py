from torch.utils.data import Dataset, DataLoader
import pandas as pd


# 一、自定义数据集
class MyDataset(Dataset):
    def __init__(self):
        file_name = 'data.xlsx'
        data = pd.read_excel(file_name)
        # 定义excel中的成员变量
        self.x1, self.x2, self.x3, self.y = data['x1'], data['x2'], data['x3'], data['y']

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.x3[index], self.y[index]

    def __len__(self):
        return len(self.x1)


# 二、初始化数据集
if __name__ == '__main__':
    my_dataset = MyDataset()
    my_dataset_loader = DataLoader(my_dataset,
                                   batch_size=2,  # 每次取值的长度
                                   shuffle=True,  # 是否打乱顺序
                                   num_workers=2  # 启用的线程数量
                                   )

    for x1, x2, x3, y in my_dataset_loader:
        print(f"x1={x1},x2={x2},x3={x3},y={y}")
