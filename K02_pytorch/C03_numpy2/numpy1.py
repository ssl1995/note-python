from PIL import Image
import numpy as np


path = './jk.png'
im = Image.open(path)

# (962, 678)
print(im.size)

# (678, 962, 3)
# Pillow 读入后通道的顺序就是 R、G、B，而 OpenCV 读入后顺序是 B、G、R。
im_pillow = np.asarray(im)
print(im_pillow.shape)

# 索引与切片
im_pillow_c1 = im_pillow[:, :, 0] # 索引与切片

im_pillow_c2 = im_pillow[:, :, 1]

im_pillow_c3 = im_pillow[:, :, 2]

print('--------------')

# (678, 962, 1)
zeros = np.zeros((im_pillow.shape[0], im_pillow.shape[1], 1))
print(zeros.shape)

# im_pillow_c1_3ch = np.concatenate((im_pillow_c1, zeros, zeros),axis=2)
