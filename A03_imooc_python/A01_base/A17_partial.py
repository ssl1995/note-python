import math
from functools import partial

# 创建一个偏函数，用于计算某个数的平方
pow2 = partial(math.pow, 2)
# 调用偏函数并打印结果
print(pow2(4))  # 输出 16


# 结合对象，进行排序，预定义采用grade进行排序
students = [
    {'name': 'Alice', 'age': 20, 'grade': 85},
    {'name': 'Bob', 'age': 22, 'grade': 92},
    {'name': 'Charlie', 'age': 21, 'grade': 78}
]

def sort_by_key(data, key):
    return sorted(data, key=lambda student: student[key])

# 创建一个偏函数，用于根据年级排序
sort_by_grade = partial(sort_by_key, key='grade')

# 对学生列表进行排序
sorted_students = sort_by_grade(students)
print(sorted_students)
