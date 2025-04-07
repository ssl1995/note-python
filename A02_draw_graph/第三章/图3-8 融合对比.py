import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
models = ['PGAHN-Image', 'PGAHN-Text', 'PGAHN（本文）']
metrics = ['准确率', '精确率', '召回率', 'F1值']
data = [
    [89.7, 88.0, 68.2, 91.5],  # PGAHN-Image
    [75.4, 73.1, 54.7, 80.3],  # PGAHN-Text
    [92.6, 91.3, 80.5, 93.8]   # PGAHN（本文）
]

# 可视化参数
bar_width = 0.25
index = np.arange(len(metrics))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 创建画布
plt.figure(figsize=(12, 6), dpi=100)

# 绘制柱状图
for i in range(len(models)):
    positions = index + i * bar_width
    bars = plt.bar(positions, data[i], bar_width,
                   color=colors[i],
                   edgecolor='black',
                   label=models[i])

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}%',
                 ha='center', va='bottom',
                 fontsize=10)

# 图表装饰
plt.xlabel('性能指标', fontsize=12)
plt.ylabel('百分比 (%)', fontsize=12)
plt.title('多模态融合精度对比', fontsize=14, pad=20)
plt.xticks(index + bar_width, metrics, rotation=0)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 坐标轴范围
plt.ylim(0, 110)

# 显示网格
plt.grid(axis='y', alpha=0.3)

# 优化布局
plt.tight_layout()
plt.show()