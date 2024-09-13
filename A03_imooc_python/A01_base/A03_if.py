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

# 可以使用elif简化多个判断条件
score = 81
if score < 60:
    print('抱歉，考试不及格')
elif score >= 90:
    print('恭喜你，拿到卓越的成绩')
elif score >= 80:
    print('恭喜你，拿到优秀的成绩')
else:
    print('恭喜你，考试及格')
