from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# NumPy下:深度学习中常用的结构
# 1.数据预加载阶段
path = './jk.png'
# Pillow:读取图片,转换成numpy需要asarray
im = Image.open(path)
im_pillow = np.asarray(im)  # asarray是浅拷贝 = 改变im_pillow会改变原始数据
print(im.size)
print(im_pillow.shape)
# Pillow 读入后通道的顺序就是 R、G、B
# OpenCV 读入后顺序是 B、G、R。

print('-----------')
# 2.索引与切片
# 取第0个维度的全部数据
im_pillow_c1 = im_pillow[:, :, 0]
print(im_pillow_c1.shape)
print('-----------')
im_pillow_c2 = im_pillow[:, :, 1]

im_pillow_c3 = im_pillow[:, :, 2]

print('--------------')

# 3.数组的拼接
# 生成一个和im等宽高的全0数组
zeros = np.zeros((im_pillow.shape[0], im_pillow.shape[1], 1))
print(zeros.shape)
# 以下会报维度错误
# im_pillow_c1_3ch = np.concatenate((im_pillow_c1, zeros, zeros),axis=2)
# (1)升级维度:np.newaxis
im_pillow_c1 = im_pillow_c1[:, :, np.newaxis]
im_pillow_c1_3ch = np.concatenate((im_pillow_c1, zeros, zeros), axis=2)
# (2)直接赋值
im_pillow_c2_3ch = np.zeros(im_pillow.shape)
im_pillow_c2_3ch[:, :, 1] = im_pillow_c2

im_pillow_c3_3ch = np.zeros(im_pillow.shape)
im_pillow_c3_3ch[:, :, 2] = im_pillow_c3

# 绘图：https://matplotlib.org/stable/gallery/index.html

plt.subplot(2, 2, 1)
plt.title('Origin Image')
plt.imshow(im_pillow)
plt.axis('off')
plt.subplot(2, 2, 2)
plt.title('Red Channel')
plt.imshow(im_pillow_c1_3ch.astype(np.uint8))
plt.axis('off')
plt.subplot(2, 2, 3)
plt.title('Green Channel')
plt.imshow(im_pillow_c2_3ch.astype(np.uint8))
plt.axis('off')
plt.subplot(2, 2, 4)
plt.title('Blue Channel')
plt.imshow(im_pillow_c3_3ch.astype(np.uint8))
plt.axis('off')
plt.savefig('./rgb_pillow.png', dpi=150)
print('-----------')

# 4.深拷贝和浅拷贝
a = np.arange(6)
print(a.shape)
print(a)
print('-----------')
b = a.view()  # asarray = 浅拷贝 = 视图
print(b.shape)
b.shape = 2, 3
print('-----------')
print(b)
print('-----------')
b[0, 0] = 111
print(b)
print('-----------')
print(a)  # 浅拷贝，a和b共享数据，b变，a也会变
print('-----------')

# 深拷贝,array
# 更加简单的方式获得三个通道的 BGR 数据，只需要将图片读入后，直接将其中的两个通道赋值为 0 即可
im_pillow = np.array(im)
im_pillow[:, :, 1:] = 0
print('-----------')

# 5.模型评估
# argMax和argMin：求最大 / 最小值对应的索引

# argsort:数组排序后返回原数组的索引
probs = np.array([0.075, 0.15, 0.075, 0.15, 0.0, 0.05, 0.05, 0.2, 0.25])
print(probs)
print('-----------')
print(np.argsort(probs))
print('-----------')
# 注意，加了负号，是按降序排序
probs_idx_sort = np.argsort(-probs)
print('-----------')

print(probs_idx_sort)
# 概率最大的前三个值的坐标
print('-----------')
print(probs_idx_sort[:3])
