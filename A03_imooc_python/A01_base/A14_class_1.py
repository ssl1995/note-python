class Base1:
    def feature1(self):
        return "Feature 1 from Base1"


class Base2:
    def feature2(self):
        return "Feature 2 from Base2"


# 多重继承
class Derived(Base1, Base2):  # 继承 Base1 和 Base2
    pass


# 创建 Derived 类的一个实例
obj = Derived()

print(obj.feature1())  # 调用 Base1 的方法
print(obj.feature2())  # 调用 Base2 的方法


class CallableObject:
    def __init__(self, value):
        self.value = value

    def __call__(self, increment):
        return self.value + increment

# 创建一个CallableObject的实例
obj = CallableObject(10)

# 调用这个对象，就像调用一个函数
result = obj(5)  # 这里传递的5将作为increment参数传递给__call__方法
print(result)  # 输出应该是15，因为 10 + 5 = 15