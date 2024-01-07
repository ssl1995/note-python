import numpy
import matplotlib.pyplot as plt
import numpy as np


# 定义画图的函数
def plot_activation_func(func, x, title):
    y = func(x)
    plt.plot(x, y)  # plot画折线图
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()


# 定义sigmod
def sigmod(x):
    return 1 / (1 + np.exp(-x))


# tanh函数
def tanh(x):
    return np.tanh(x)


# relu
def relu(x):
    return np.maximum(0, x)


# leaky_relu
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)


# parametric_relu
def parametric_relu(x, alpha=0.5):
    return np.where(x > 0, x, x * alpha)


# parametric_relu
def elu(x, alpha=1):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


x_value = np.arange(-10, 10, 0.1)
plot_activation_func(elu, x_value, "sigmod")
