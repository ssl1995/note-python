import os
import cv2 as cv
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np


class MyImageDataset(Dataset):
    def __init__(self):
        image_root = r"./animals"
        self.file_path_list = []
        dir_name = []
        self.labels = []
        for root, dirs, files in os.walk(image_root):
            if dirs:
                dir_name = dirs
            for file_i in files:
                file_i_full_path = os.path.join(root, file_i)
                self.file_path_list.append(file_i_full_path)

                label = root.split(os.sep)[-1]
                label_id = dir_name.index(label)
                self.labels.append(label_id)

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, item):
        img = cv.imread(self.file_path_list[item])
        # 归一化
        img = cv.resize(img, dsize=(256, 256))
        # HWC
        # 012
        # CHW
        # 210
        img = np.transpose(img, (2, 1, 0))
        img_tensor = torch.from_numpy(img)
        label = self.labels[item]
        return img_tensor, label


if __name__ == '__main__':
    my_image_dataset = MyImageDataset()
    my_dataloader = DataLoader(my_image_dataset, batch_size=4)
    for x_i, y_i in my_dataloader:
        print(x_i.shape, y_i)

# image_root = r"./animals"
# for root, dirs, files in os.walk(image_root):
#     for file_i in files:
#         file_i_full_path = os.path.join(root, file_i)
#         print("文件路径：", file_i_full_path)
#
#         label = root.split(os.sep)[-1]
#         print("文件label：", file_i_full_path)
