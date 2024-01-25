import torchvision.models as models
from PIL import Image
import torchvision
import torchvision.transforms as transforms

alexnet = models.alexnet(pretrained=True)

im = Image.open('img.png')

transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(im).unsqueeze(0)
alexnet(input_tensor).argmax()

# 模型的微调，需要打印出输出的结构
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               target_transform=None,
                                               download=True)
# 取32张图片的tensor
tensor_dataloader = DataLoader(dataset=cifar10_dataset,
                               batch_size=32)
data_iter = iter(tensor_dataloader)
img_tensor, label_tensor = data_iter.next()
print(img_tensor.shape)
grid_tensor = torchvision.utils.make_grid(img_tensor, nrow=16, padding=2)
grid_img = transforms.ToPILImage()(grid_tensor)
print(grid_img)
