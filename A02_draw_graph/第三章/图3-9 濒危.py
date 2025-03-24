import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（Windows系统用SimHei，Mac用Arial Unicode MS）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换成系统中已安装的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据预处理
models = ['PGAHN(ours)', 'ResNet-50', 'EfficientNet-B4', 'ViT-Base', 'YOLOv7']
metrics = ['准确率', '精确率', '召回率', 'F1值']

# 将百分比转换为浮点数
data = np.array([
    [89.1, 87.9, 84.8, 86.2],  # PGAHN
    [75.9, 77.2, 64.9, 70.3],  # ResNet-50
    [79.0, 80.5, 69.0, 74.0],  # EfficientNet-B4
    [81.2, 81.9, 71.2, 76.1],  # ViT-Base
    [83.3, 81.1, 76.3, 78.5]   # YOLOv7
])

# 可视化参数
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']  # 青绿/蓝/橙/紫
x = np.arange(len(models))  # 模型坐标
width = 0.2  # 柱宽
indices = np.arange(len(metrics))  # 指标数量

# 创建画布
plt.figure(figsize=(14, 7))

# 绘制分组柱状图
for i in range(len(metrics)):
    offset = width * (i - 1.5)  # 计算偏移量
    bars = plt.bar(x + offset, data[:, i], width,
                  color=colors[i],
                  label=metrics[i],
                  edgecolor='white',
                  linewidth=0.7)

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=10, color='black')

# 坐标轴设置
plt.xticks(x, models, rotation=10, fontsize=12)
plt.ylabel('性能指标 (%)', fontsize=12)
plt.ylim(60, 95)  # 聚焦主要数据区间
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 图例和标题
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=12)
plt.title('濒危树种测试集精度模型对比柱状图', fontsize=16, pad=20, fontweight='bold')

# 显示图表
plt.tight_layout()
plt.show()