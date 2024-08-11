from PIL import Image
from torchvision import transforms
import torch

# 一、图片转Tensor
# 字符串前的 r 表示该字符串是一个 原始字符串（raw string）。
# 原始字符串的特点是其中的 转义字符 不会被解释器解析，而是被视为普通的字符。
image_path = r'sishen.jpg'

image = Image.open(image_path)
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=319x180 at 0x1B8D9DB0E50>
print(image)

tensor_image = transforms.ToTensor()(image)
# <class 'torch.Tensor'>
print(type(tensor_image))

# 二、tensor转图片
tensor_image = torch.randn((3, 244, 244))

transformed_image = transforms.ToPILImage()(tensor_image)

save_path = r'save_image.jpg'

transformed_image.save(save_path)
