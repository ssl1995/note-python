import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
models = ['PGAHN(本文)', 'ResNet-50', 'EfficientNet-B4', 'ViT-Base', 'YOLOv7']
metrics = {
    '参数量（M）': [18.7, 25.6, 19.3, 86.6, 36.9],
    'FLOP（S）': [4.3, 7.6, 4.9, 16.2, 14.2],
    '推理速度（FPS）': [41, 32, 28, 19, 63],
    '显存占用（G）': [5.2, 6.8, 5.9, 9.7, 3.1]
}

# 创建子图矩阵
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 绘制每个指标的柱状图
for ax, (metric, values) in zip(axs.flat, metrics.items()):
    bars = ax.bar(models, values, color=colors, edgecolor='black')
    ax.set_title(metric, fontsize=12, pad=10)
    ax.set_ylabel(metric.split('（')[0], fontsize=10)
    ax.tick_params(axis='x', rotation=30)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom',
                fontsize=9)

# 调整布局
plt.tight_layout(pad=3.0)
plt.show()