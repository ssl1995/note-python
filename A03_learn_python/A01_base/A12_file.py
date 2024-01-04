'''
文件操作
'''
import shutil
import os

# 复制当个文件
# shutil.copyfile()

# 复制文件目录
# shutil.copytree()

# 移动文件
# shutil.move(a, b)

# 删除文件
# os.remove(c)

'''
文件夹操作
'''

dir_name = "my_dir"
if os.path.exists(dir_name):
    print("文件夹已经存在")
else:
    os.mkdir(dir_name)
    print("文件夹已经新建成功")

# 遍历文件夹
root_dir = dir_name

file_full_path_list = []
for root, dirs, files in os.walk(root_dir):
    for file_i in files:
        file_i_full_path = os.path.join(root, file_i)
        file_full_path_list.append(file_i_full_path)

print(file_full_path_list)

# 删除空文件夹
if (os.path.exists(dir_name)):
    os.rmdir(dir_name)
else:
    os.mkdir(dir_name)

# 删除非空文件，文件夹里有东西，也删除
if (os.path.exists(dir_name)):
    shutil.rmtree(dir_name)
else:
    os.mkdir(dir_name)
