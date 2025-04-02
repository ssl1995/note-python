import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 数据定义
models = ['XGBoost', 'TextCNN', 'BiLSTM', 'CNN-LSTM', 'CharLSTM', 'Transformer', 'MPFE-Net', 'SHT（Ours）']
accuracy = [72.34, 79.56, 80.92, 82.17, 83.41, 84.25, 85.73, 93.85]
precision = [68.21, 76.83, 78.45, 80.19, 81.05, 82.10, 83.92, 92.67]
recall = [73.15, 81.02, 82.13, 83.76, 84.32, 85.03, 86.05, 94.23]
f1 = [70.12, 78.64, 80.18, 81.87, 82.62, 83.52, 84.96, 93.44]

# 可视化设置
bar_width = 0.2  # 单柱宽度
x = np.arange(len(models))  # x轴基准位置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 四组颜色

# 创建画布
plt.figure(figsize=(15, 6), dpi=100)

# 绘制四组柱状图
bars1 = plt.bar(x - 1.5*bar_width, accuracy, bar_width, color=colors[0], edgecolor='black', label='准确率')
bars2 = plt.bar(x - 0.5*bar_width, precision, bar_width, color=colors[1], edgecolor='black', label='精确率')
bars3 = plt.bar(x + 0.5*bar_width, recall, bar_width, color=colors[2], edgecolor='black', label='召回率')
bars4 = plt.bar(x + 1.5*bar_width, f1, bar_width, color=colors[3], edgecolor='black', label='F1值')

# 坐标轴与标题
plt.xticks(x, models, fontsize=10, rotation=15)
plt.ylabel("性能指标 (%)", fontsize=12)
plt.title("不同模型性能指标对比分析", fontsize=14, pad=20)

# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# 图例与布局优化
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 右侧图例
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 显示图形
plt.show()