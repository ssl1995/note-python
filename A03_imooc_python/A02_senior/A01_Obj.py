class NeuralNetwork:
    # 构造器
    def __init__(self, init_w, init_b):
        self.w = init_w
        self.b = init_b

    def forward(self, x):
        # 函数功能的封装
        y = self.w * x + self.b
        return y

    def show_param(self):
        print("我的神经网络参数如下:")
        print(f"self.w = {self.w}")
        print(f"self.b = {self.b}")


ne_1 = NeuralNetwork(2, 3)

ne_1.show_param()


# 继承、也可以是多态
class NeuralNetwork2(NeuralNetwork):
    def __init__(self, init_c, init_w, init_b):
        super().__init__(init_w, init_b)
        self.c = init_c

    def printC(self):
        print(f"我的c={self.c}")


network2 = NeuralNetwork2(1, 2, 3)
network2.printC()
