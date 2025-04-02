import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 数据预处理
models = ['PGAHN(本文)', 'ResNet-50', 'EfficientNet-B4', 'ViT-Base', 'YOLOv7']
metrics = ['准确率', '精确率', '召回率', 'F1值']
data = np.array([
    [92.3, 90.8, 91.5, 91.9],  # PGAHN
    [82.1, 83.5, 71.2, 76.8],  # ResNet-50
    [85.6, 86.9, 75.4, 80.2],  # EfficientNet-B4
    [87.9, 88.3, 77.8, 82.4],  # ViT-Base
    [89.5, 87.2, 83.1, 84.9]  # YOLOv7
])

# 可视化设置
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # 指标颜色
x = np.arange(len(models))  # 模型位置
width = 0.2  # 柱状图宽度

# 创建画布
plt.figure(figsize=(14, 8), dpi=100)
ax = plt.gca()

# 绘制分组柱状图
for i in range(len(metrics)):
    offset = width * (i - 1.5)  # 计算每组柱子的偏移量
    bars = ax.bar(x + offset, data[:, i], width,
                  color=colors[i], edgecolor='white',
                  label=metrics[i], alpha=0.9)

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=9, color='black')

# 坐标轴设置
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_ylim(60, 100)  # 突出差异区间
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例和标题
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.title('INATURALIST数据集精度对比',
          fontsize=14, pad=20, fontweight='bold')

# 显示图表
plt.tight_layout()
plt.show()
