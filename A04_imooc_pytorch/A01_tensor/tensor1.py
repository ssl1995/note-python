import torch

# 1、使用cpu还是显卡
device = "cuda" if torch.cuda.is_available() else "cpu"

shape = [1,2,]

tensor_1 = torch.tensor(shape)

print("tensor_1:",tensor_1)

tensor_1 = tensor_1.to(device)

print("使用的设备:",tensor_1.device)
