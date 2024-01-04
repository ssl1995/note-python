'''
循环结构
for\
while
'''

x = 5
for x_i in range(x):
    print("x_i=", x_i)

ops = ['a', 'v', 'c', 'd']
for op in ops:
    print("op=", op)

for index, op in enumerate(ops):
    print(f"index={index},op={op}")

isFind = False

if not isFind:  # py中的取反是 not
    print("没有发现")
else:
    print("发现啦")

a = 10
b = 20
result = a + b

while True:
    inputNumber = int(input(f"请输入{a}+{b}的结果:\n"))
    if inputNumber == result:
        print("回答对")
        break
    else:
        print("回答错")
