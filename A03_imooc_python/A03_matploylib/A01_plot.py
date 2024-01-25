import pandas as pd
import matplotlib.pyplot as plt

# 定位文件
file_1_loc = 'resources/yolov5l.csv'
file_2_loc = 'resources/yolov5m.csv'
file_3_loc = 'resources/yolov5s.csv'

# 读取文件
file_1 = pd.read_csv(file_1_loc)
file_2 = pd.read_csv(file_2_loc)
file_3 = pd.read_csv(file_3_loc)

# 横坐标
file_1_train_box_loss = file_1['      train/box_loss']
file_2_train_box_loss = file_2['      train/box_loss']
file_3_train_box_loss = file_3['      train/box_loss']

# 纵坐标
x_list = [i for i in range(len(file_1_train_box_loss))]

# 画多个折线图
plt.plot(x_list, file_1_train_box_loss)
plt.plot(x_list, file_2_train_box_loss)
plt.plot(x_list, file_3_train_box_loss)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train box_loss")
plt.grid() #填充网格

plt.legend(['yolov5l', 'yolov5m', 'yolov5s']) #展示右上角的标题

plt.show() # 展示图片


for i in range(10):
    print(i)