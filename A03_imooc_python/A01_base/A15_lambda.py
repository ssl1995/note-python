# 函数式编程：将一个函数作用到一个数据类型上
# map()
import math
from functools import reduce

def f2(x):
    return x * x

nums = [1, 2, 3, 4]
# for num in map(f2, nums):
#     print(num)


# reduce(),接受2个参数
def f3(x, y):
    return x * y


nums = [1, 2, 3, 4]
res = reduce(f3, nums)
print(res)


# filter()
def f4(x):
    num = int(math.sqrt(x))
    return num * num == x


nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
res = filter(f4, nums)
for var in res:
    print(var)

# sorted()排序
score = [('Alice', 72), ('Candy', 90), ('Bob', 62)]


def sort_fun(x):
    return x[1]  # ==> 按成绩排序，成绩是第二个字段


new_score = sorted(score, key=sort_fun, reverse=True)
print(score)
print(new_score)


# 闭包：内部函数引用外部已经结束函数的操作
def counter():
    count = [0]  # 使用列表使得 count 在 inner 中可变

    def increment():
        count[0] += 1
        return count[0]

    return increment


# 创建一个计数器闭包
count_up = counter()

# 使用闭包递增计数器
print(count_up())  # 输出 1
print(count_up())  # 输出 2
print(count_up())  # 输出 3

# 匿名函数lambda 输入:输出
numbers = [1, 2, 3, 4, 5, 6]
# sorted
new_score1 = sorted(score, key=lambda x: x[1], reverse=True)
print(new_score1)  # 输出 [1, 4, 9, 16, 25, 36]

# 使用map()和lambda函数计算每个数字的平方
squares = map(lambda x: x ** 2, numbers)
print(list(squares))  # 输出 [1, 4, 9, 16, 25, 36]

# 使用filter()和lambda函数过滤出偶数
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # 输出 [2, 4, 6]
