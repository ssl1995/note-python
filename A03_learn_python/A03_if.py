'''
选择结构
if
'''
x = 5
if x > 10:
    print("x=", x)
elif x == 5:
    print("x=", x)
else:
    print("x=", x)

a = 10
b = 20
result = a + b
inputNumber = int(input(f"请输入{a}+{b}的结果:\n"))

if result == inputNumber:
    print("回答正确")
else:
    print("回答错误")
