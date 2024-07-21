import numpy as np
import matplotlib.pyplot as plt

# 输入数据
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# 标签
Y = np.array([[0, 1, 1, 0]])
# 第一层网络层参数矩阵，初始化输入层权值，取值范围为-1到1
# np.random.random生成一个形状为(3,4)的二维数组，数组中的每个元素都是0到1之间的随机数。
V = (np.random.random((3, 4)) - 0.5) * 2
# 第二层网络层参数矩阵，初始化输出层权值，取值范围为-1到1
W = (np.random.random((4, 1)) - 0.5) * 2


# 可视化
def get_show():
    # 正样本
    all_positive_x = [0, 1]
    all_positive_y = [0, 1]
    # 负样本
    all_negative_x = [0, 1]
    all_negative_y = [1, 0]

    plt.figure()
    plt.plot(all_positive_x, all_positive_y, 'bo')
    plt.plot(all_negative_x, all_negative_y, 'yo')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# get_show()


lr = 0.11  # 学习速率


# 激活函数，输出0到1
def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


# 激活函数的倒数
def dsigmoid(x):
    x = x * (1 - x)
    return x


def update():
    global X, Y, W, V, lr
    # 第一层网络：输入*第一层网络参数矩阵
    L1 = sigmoid(np.dot(X, V))
    # 第二层网络：第一层输出*第二层网络参数矩阵
    L2 = sigmoid(np.dot(L1, W))
    # 输出层的误差 = 下一层的误差*当前层激活函数的导数*与下一层的连接权重矩阵
    L2_delata = (Y.T - L2) * dsigmoid(L2)
    # 隐藏层的误差=下一层的误差*当前层激活函数的导数*与下一层的连接权重矩阵
    L1_delata = L2_delata.dot(W.T) * dsigmoid(L1)

    W_C = lr * L1.T.dot(L2_delata)
    V_C = lr * X.T.dot(L1_delata)

    W = W + W_C
    V = V + V_C


errors = []  # 记录误差

for i in range(10000):
    update()
    if i % 1000 == 0:
        L1 = sigmoid(np.dot(X, V))
        L2 = sigmoid(np.dot(L1, W))

        errors.append(np.mean(np.abs(Y.T - L2)))  # np.mean是求平均值
        print('Error:', np.mean(np.abs(Y.T - L2)))

plt.plot(errors)
plt.ylabel('errors')
plt.show()

L1 = sigmoid(np.dot(X, V))  # 隐藏层输出(4*3)*(3*4)=(4,4)
L2 = sigmoid(np.dot(L1, W))  # 输出层输出(4,4)*(4*1)=(4,1)

print(L2)


def classify(x):
    if x > 0.5:
        return 1
    else:
        return 0


for i in map(classify, L2):  # L2一共四个数
    print(i)
