'''
set()
py中的集合具有去重的功能
'''
x = {1, 23, 23}
print(x)

y = set()
y.add(1)
y.add(1)
y.add(2)
print(y)

# 交集、并集、差集
set1={1,2,3,4}
set2={3,4,5,6}

print(set1.intersection(set2))
print(set1.union(set2))
print(set1.difference(set2))
