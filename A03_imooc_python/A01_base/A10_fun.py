'''
函数
参数传递
    使用关键字传参，形参顺序就可以变
    定义函数的时候，可以设置默认值
    可变参数，可以接收到多个默认值
变量作用域
    全局函数、局部函数
匿名函数
综合案例
'''


# 参数传递：可变参数
def fun1(model, *layers):
    for index, layer in enumerate(layers):
        print(f"Adding layer {layer} to model {model}.")


fun1("model", "conv", "rele", "softmax")

# 匿名函数：前面是输入、后面是输出
fun_units = lambda layer: layer * 128

res = fun_units(10)

print(res)
