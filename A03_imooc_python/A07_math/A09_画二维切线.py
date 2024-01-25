import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 2


def df(x):
    return 2 * x


# 画出从-3到3的图像
x = np.linspace(-3, 3, 100)
y = f(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="f(x)=x^2")

# 求在x=1时的切点（梯度）和切线
x1 = 1
y1 = f(x1)
slope = df(x1)


#  定义切线方程：k(x-x1)+y1
def tangent_line(x, x1, y1, slope):
    return slope * (x - x1) + y1

x_tangent = np.linspace(x1 - 1, x1 + 1, 10)
y_tangent = tangent_line(x_tangent, x1, y1, slope)

plt.plot(x_tangent, y_tangent, label="Tangent at x = 1", color='red')
plt.scatter([x1], [y1], color='black')

plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Function and Tangent Line at a Point")
plt.grid(True)
plt.show()
