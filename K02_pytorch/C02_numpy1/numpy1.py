import numpy as np
import matplotlib.pyplot as plt

# numpy上
# 1.创建数组
# asarray是浅拷贝;array是深拷贝
arr_1_d = np.asarray([1])
print(arr_1_d)

arr_2_d = np.asarray([[1, 2], [3, 4]])
print(arr_2_d)

# 2.数组的属性
# 数组的维度、形状、size和数据类型
# ndim shape
print(arr_2_d.shape)  # (2, 2)：理解为2个2维数据

arr_3_d = np.asarray([[1, 2], [3, 4], [5, 6]])
print(arr_3_d.shape)  # (3, 2)：理解为3个2维数据

# 转换数组的形状的时候，需要保证元素个数相同
arr_2_d_re = arr_2_d.reshape((4, 1))
print(arr_2_d_re)

print('--------')
# 可以使用np.reshape，可以指定行优先、列优先、原数组方式
a = np.arange(6).reshape(2, 3)  # 2行3列
print(a)

a.reshape(3, 2)  # 默认按照行优先转成3行2列
print(a)

a.reshape(3, 2, order='F')  # 按照列优先转成3行2列
print(a)

# size = 行数 × 列数
print(a.size)

# dtype = 描述数组中元素类型的对象
# 可以使用astype()改变数组的数据类型，但是需要注意的是，astype()会创建一个新的数组，不会改变原数组
print(a.dtype)

# 3.其他生成数组的工具 = ones、zeros、arange()、linspace()
# 使用场景：比如默认生成2*3的0.5的权重
# [[1. 1. 1.] 1.这样是默认没有指定数据类型
#  [1. 1. 1.]]
ones = np.ones(shape=(2, 3))
print(ones)

print(np.ones((2, 3)) * 0.5)

# arange()函数，可以用来生成图片的x轴、y轴坐标
# arange([start,] stop[, step,], dtype=None, *, like=None)
print(np.arange(5))  # [0 1 2 3 4]

print(np.arange(1, 5))  # [1 2 3 4]

print(np.arange(1, 8, 3))  # [1 4 7]

# linspace()函数:从2开始，10结束，有3个数的等差数列
lin_space = np.linspace(start=2, stop=10, num=3)  # [ 2.  6. 10.]
print(lin_space)

# 应用：画图Y=X^2
X = np.arange(-50, 51, 2)
Y = X ** 2
plt.plot(X, Y, color='blue')
plt.legend()
plt.show()

# 4 数组的轴
a = np.arange(18).reshape(3, 2, 3)
# [[[ 0  1  2]
#   [ 3  4  5]]
#
#  [[ 6  7  8]
#   [ 9 10 11]]
#
#  [[12 13 14]
#   [15 16 17]]]
# [[12 13 14]
#  [15 16 17]]
print(a)

# 当axis=0时，表示沿着第0轴进行操作，即对每一列进行操作
# 当axis=i时，表示沿着第i轴进行操作，可以理解为第i个轴的数据将被折叠或聚合一起
print(a.max(axis=0))
print('-----------')
print(a.max(axis=1))
print('-----------')
print(a.max(axis=2))
print('-----------')
