'''
序列索引
py支持负数
num[-1]: 从后王前数
'''
numbers = [10, 11, 12, 13, 14]
# 左闭右开
print(numbers[0:2])
print(numbers[:2])

print(numbers[:-2])
# 步长是第三个数：为2
print(numbers[0:4:2])

'''
序列相加和增删改查
'''
a = [1, 2, 3]
a.append(4)
# 删除第一个元素，这里从[1 -> size]
a.remove(1)
b = ["b1", "b2", 4]
c = a + b
# [1, 2, 3, 'b1', 'b2', 4]
print(c)

'''
序列相乘
'''
a = [1,2,3]
b = a*3
# [1, 2, 3, 1, 2, 3, 1, 2, 3]
print(b)

'''
列表的排序
'''
a = [23,23,15,123,124,125]
# 排序完后，不需要返回值
a.sort()
print(a)
# 降序排序
a.sort(reverse=True)
print(a)

# 快速创建一个有序列表
x_list =[ i for i in range(10) ]
print(x_list)



