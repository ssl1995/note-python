import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ================== 字体全局设置 ==================
rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
rcParams['font.size'] = 26  # 全局字体大小调整为16

# ================== 数据准备 ======================
models = ['PGAHN(本文)', 'ResNet-50', 'EfficientNet-B4', 'ViT-Base', 'YOLOv7']
metrics = ['准确率', '精确率', '召回率', 'F1值']
data = np.array([
    [92.3, 90.8, 91.5, 91.9],  # PGAHN
    [82.1, 83.5, 71.2, 76.8],  # ResNet-50
    [85.6, 86.9, 75.4, 80.2],  # EfficientNet-B4
    [87.9, 88.3, 77.8, 82.4],  # ViT-Base
    [89.5, 87.2, 83.1, 84.9]  # YOLOv7
])

# ================== 可视化设置 ====================
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # 新配色方案
x = np.arange(len(models))  # 模型位置坐标
width = 0.2  # 柱宽

# ================== 创建画布 ======================
plt.figure(figsize=(18, 10), dpi=200)  # 增大画布尺寸
ax = plt.gca()

# ================== 绘制分组柱状图 ================
for i in range(len(metrics)):
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, data[:, i], width,
                  color=colors[i], edgecolor='white',
                  label=metrics[i], alpha=0.9)

    # 数据标签（字体放大到14）
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=17, color='black')  # 字体从9→14

# ================== 坐标轴设置 ====================
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=26)  # X轴标签字体从12→16
ax.set_ylabel('评价指标 (%)', fontsize=26, labelpad=15)
ax.set_ylim(60, 100)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# ================== 图例与标题 ==================
plt.legend(loc='upper left',
           bbox_to_anchor=(1, 1),
           frameon=False,
           fontsize=26)  # 图例字体放大
plt.title('INATURALIST数据集精度对比',
          fontsize=26,  # 标题字体从14→20
          pad=25,  # 增加标题间距
          fontweight='bold')  # 加粗标题

# ================== 显示图表 =====================
plt.tight_layout()
plt.show()
