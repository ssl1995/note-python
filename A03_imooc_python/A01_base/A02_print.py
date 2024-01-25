'''
变量赋值：
1.基本赋值
2.同一个值给多个变量
3.多重辅助
4.使用下划线的赋值,py中某些函数的返回值是使用不到的 所以使用下划线
'''
x,_ =1,2

print(x)

'''
屏幕输出
'''
print("Hello","World",sep='-')
print("Hello","World",end='!\n')
'''
键盘输入
'''
# input = input("用户请输入：")
# print("用户输入的值:",input)

'''
查看数据类型：type(x)
数值转换：
float()
str()
'''
y = 3.13
print(type(y))

name ="Alen"
age = 30
print("My name is %s,age is %d."%(name,age))
print("My name is {},age is {}.".format(name,age))
print(f"My name is {name},age is {age}.") # 推荐这种方式格式化

'''
控制精度
'''
number = 2.123123
print("精度:%.2f"%(number))
print("精度:{:.2f}".format(number))
print(f"精度:{number:.2f}")


