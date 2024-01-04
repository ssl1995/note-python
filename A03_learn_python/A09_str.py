'''
字符串
py中的集合具有去重的功能
'''
str  ="this_is_a_file.jpg"
# 获取整个字符串
str1 = str[:]
print(str1)

# 字符串最后3个字符
str2= str[-3:]
print(str2)

# 获取字符串的第2到倒数第3个字符，每隔2个字符取一个
subStr = str[1:-2:2]
print(subStr)

# 反转字符串
reversStr = str[::-1]
print(reversStr)
