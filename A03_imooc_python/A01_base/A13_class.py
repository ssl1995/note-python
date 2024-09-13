# 定义一个类
class Person(object):
    # 可以修改的属性，除此之外的不允许修改
    __slots__ = ('name', 'age')

    def __init__(self, name, age):
        self.name = name
        self.age = age

    pass


person1 = Person("张三", 19)
person1.name = "张三1"  # 不允许
print(person1.name)

# 创建一个Person实例
p = Person('Alice', 30)

# 尝试动态添加一个新属性将会失败
try:
    p.name = 'female'
except AttributeError as e:
    print(e)
