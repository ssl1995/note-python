import numpy as np
import matplotlib.pyplot as plt


# 定义一个凸函数
def convex_function(x):
    return x ** 2 + 3 * x + 2


# 创建x值范围
x = np.linspace(-10, 10, 400)
y = convex_function(x)

# 绘制图形
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='f(x) = x^2 + 3x + 2')
plt.title('Convex Function Example')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
