# 装饰器，使用@修饰一个函数
# 无参的装饰器
def simple_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")

    return wrapper


@simple_decorator
def say_hello():
    print("Hello!")


say_hello()


# 有参的装饰器
def repeat(num_times):
    # 外部定义了次数，内部就需要定一个func
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator_repeat


@repeat(3)
def greet(name):
    print(f"Hello {name}!")


greet("World")


# 装饰器处理带参数的函数
def log_decorator(func):
    # func有参数，所以需要*args和**kwargs
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}' with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)

    return wrapper


@log_decorator
def add(a, b):
    return a + b


result = add(10, 20)
print(result)  # 输出 30
print(add.__name__)  # 丢失了原函数名称，
print(add.__dict__)  # 丢失了原函数文档字符串

from functools import wraps


# 使用 functools.wraps 保持函数元数据
def log_decorator1(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}' with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)

    return wrapper


@log_decorator1
def add(a, b):
    """定义一个文档字符串Adds two numbers"""
    return a + b


print(add.__name__)  # 输出 'add'
print(add.__doc__)  # 输出 'Adds two numbers'
