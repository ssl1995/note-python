
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 数据定义
models = ['PGAHN(本文)', 'ResNet-50', 'EfficientNet-B4', 'ViT-Base', 'YOLOv7']
metrics = {
    '参数量(M)': [15.1, 26.6, 21.3, 56.6, 46.9],
    'FLOP(G)': [3.1, 6.8, 5.9, 19.5, 16.2],
    '推理速度(FPS)': [35, 32, 29, 40, 41],
    '显存占用(GB)': [6.3, 6.0, 5.1, 9.6, 3.2]
}

# 可视化设置
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # 四组颜色
x = np.arange(len(models))  # 模型位置
width = 0.2  # 柱状图宽度

# 创建 2x2 子图布局
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('INATURALIST数据集模型性能柱状图', fontsize=16)  # 修改主标题为中文

# 绘制每个指标的柱状图
for idx, (metric_name, values) in enumerate(metrics.items()):
    row, col = idx // 2, idx % 2
    ax = axs[row, col]

    bars = ax.bar(x, values, width, color=colors[idx], edgecolor='black')

    # 标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    # 坐标轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_title(metric_name, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给主标题留空间
plt.show()