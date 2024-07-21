import numpy as np
import matplotlib.pyplot as plt

n = 100  # 迭代此时
lr = 0.10  # 学习速率

# 输入数据，三个维度分别是偏置、x坐标、y坐标
X = np.array([[1, 2, 3],
              [1, 4, 5],
              [1, 1, 1],
              [1, 5, 3],
              [1, 0, 1]])
# 便签
Y = np.array([1, 1, -1, 1, -1])

# 要学习的模型：f(WX+B)，
# 结论：对于正样本，f>0;对于负样本，f<0

# 权重W初始化，取值范围-1到1
W = (np.random.random(X.shape[1]) - 0.5) * 2


# 可视化
def get_show():
    # 正样本
    all_positive_x = [2, 4, 5]
    all_positive_y = [3, 5, 3]
    # 负样本
    all_negative_x = [1, 0]
    all_negative_y = [1, 1]
    # 计算分界线斜率与截距
    k = -W[1] / W[2]
    b = -W[0] / W[2]
    # 生成x刻度
    xdata = np.linspace(0, 5)
    plt.figure()
    plt.plot(xdata, xdata * k + b, 'r')
    plt.plot(all_positive_x, all_positive_y, 'bo')
    plt.plot(all_negative_x, all_negative_y, 'yo')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# 更新权重函数 - 梯度反向传播
# 用于更新权重 W 的函数。根据学习速率 lr 和当前的权重 W，通过梯度反向传播更新权重，以使模型更好地拟合数据
def get_update():
    global X, Y, W, lr, n
    n += 1
    new_output = np.sign(np.dot(X, W.T))

    new_W = W + lr * ((Y - new_output.T).dot(X))

    W = new_W


if __name__ == '__main__':
    W = (np.random.random(X.shape[1]) - 0.5) * 2
    print("原始权重：", W)
    print("原始权重.T：", W.T)  # W.T是将W矩阵转置


# 通过循环调用 get_update() 更新权重，直到模型成功将所有样本分类正确为止。最后调用 get_show() 可视化分类结果。
def main():
    for _ in range(n):
        get_update()
        new_output = np.sign(np.dot(X, W.T))
        if (new_output == Y.T).all():
            print("迭代次数：", n)
            break
    get_show()
