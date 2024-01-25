import pandas as pd

# 读取csv文件
# data_loc = r'./resources/yolov5s.csv'
# data = pd.read_csv(data_loc)
# print(data)


# 读取xlsx文件
data_loc = r'./resources/销售数据.xlsx'
data = pd.read_excel(data_loc)
# 打印阅读
# print(data)

# 表信息
# print(data.describe())

# 表头
# print(data.head())

# 尾几行
# print(data.tail())

# 第一行信息
# data_0 = data.loc[0]
# print(data_0)

# 指定列
# data_row_x = data.loc[:, "大类编码"]
# data_row_1 = data.loc[1, "大类编码"]
# data_row_1_3 = data.loc[1:3, "大类编码"]

# 提取符合要求的元素
# data.loc[data["销售数量"]]

# 分组和提取符合要求的元素
# data_5 = data.loc[data["销售数量"] > 10, ["小类编码", "小类名称"]]
# print(data_5)

# 数据分组和排序
data_extract = data.groupby('商品类型')['销售金额'].sum()
# 重新排序
data_extract = data_extract.reset_index()

print(data_extract)

data_extract.to_csv("处理好的表格.csv",encoding='utf-8',index=False)
data_extract.to_excel("处理好的表格.xlsx",engine='utf-8',index=False)