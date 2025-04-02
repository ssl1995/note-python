import matplotlib.pyplot as plt
import numpy as np

# 配置中文字体 (Windows使用SimHei，Mac使用Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据准备
models = [
    'PNAS(Baseline)',
    '+Focal Loss',
    '+Focal Loss+GHM',
    '+Focal Loss+GHM+CBAM',
    'PGAHN'
]
metrics = ['准确率', '精确率', '召回率', 'F1值']

# 数据矩阵 (去除百分号后的数值)
data = np.array([
    [76.5, 79.2, 63.4, 43.2],
    [81.1, 82.7, 72.8, 57.8],
    [83.4, 85.1, 76.2, 62.1],
    [89.2, 90.6, 84.7, 73.6],
    [93.7, 92.5, 89.1, 85.4]
])

# 可视化参数
# colors = ['#4E79A7', '#59A14F', '#EDC948', '#F28E2B']  # 学术风格配色
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # 指标颜色
x = np.arange(len(models))  # 模型位置坐标
width = 0.18  # 柱状图宽度

# 创建画布
fig, ax = plt.subplots(figsize=(14, 7))

# 绘制分组柱状图
for i, metric in enumerate(metrics):
    offset = width * (i - 1.5)  # 计算每个指标的水平偏移量
    bars = ax.bar(x + offset, data[:, i], width,
                 color=colors[i],
                 label=metric,
                 edgecolor='white',
                 linewidth=0.8)

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=10, color='black')

# 坐标轴装饰
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=12)
ax.set_ylabel('性能百分比 (%)', fontsize=12)
ax.set_ylim(30, 100)  # 扩展显示范围以包含F1值
ax.yaxis.grid(True, linestyle='--', alpha=0.6)

# 添加图例和标题
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=12)
plt.title('消融实验柱状图', fontsize=16, pad=20, fontweight='bold')

# 显示图表
plt.tight_layout()
plt.show()