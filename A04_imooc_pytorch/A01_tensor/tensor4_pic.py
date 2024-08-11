from PIL import Image
from torchvision import transforms
import torch

open_path = r'sishen.jpg'
save_path = r'sishen_tensor.jpg'

# 一、获取图片
image = Image.open(open_path)

tensor_image = transforms.ToTensor()(image)
# torch.Size([3, 180, 319])
print(tensor_image)
print(tensor_image.shape)
print(tensor_image.device)

# 放到显卡中
if torch.cuda.is_available():
    tensor_image = tensor_image.to("cuda")

print(tensor_image.device)

# 每个像素+0.1
tensor_image += 0.1

# 放回内存
tensor_image = tensor_image.to('cpu')
print(tensor_image.device)

# 二、保存图片
save_image = transforms.ToPILImage()(tensor_image)
save_image.save(save_path)
