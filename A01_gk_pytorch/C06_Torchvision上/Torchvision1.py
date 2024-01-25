# 以MNIST为例
import torchvision

# torchvision: 是常用数据集 + 常见网络模型 + 常用图像处理方法。支持许多数据集
# 以MNIST为例，第一次运行，本地生成该数据集的图片、标签、测试数据
# PIL是torchvision的官方图像加载器
mnist_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=None,
    target_transform=None,
    download=True)
# 数据预览：转换列表。为两个元组：一个是图像数据；一个是图像的标签
mnist_dataset_list = list(mnist_dataset)
# print(mnist_dataset_list)
# (<PIL.Image.Image image mode=L size=28x28 at 0x1CDAE8F8D90>, 3)
one_data = mnist_dataset_list[0][0]
print(one_data)

display(mnist_dataset_list[0][0])