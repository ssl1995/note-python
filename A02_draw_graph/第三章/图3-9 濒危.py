import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
threat_types = ['良性URL', '网页篡改类URL', '钓鱼攻击URL', '恶意软件分发URL']
sample_counts = [428103, 96457, 94111, 32520]
colors = ['#4CAF50', '#FFC107', '#E53935', '#1E88E5']

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# 绘制柱状图
bars = ax.bar(threat_types, sample_counts,
             color=colors, edgecolor='black',
             width=0.6)

# 设置坐标轴
ax.set_ylabel('样本数量', fontsize=12)
ax.set_yscale('log')  # 保持对数坐标轴
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
            f'{height:,}',
            ha='center', va='bottom',
            fontsize=10)

# 设置标题和布局
plt.title('恶意URL数据集分布', fontsize=14, pad=20)
plt.xticks(rotation=15)  # X轴标签旋转15度
plt.tight_layout()

# 保存和显示
plt.savefig('malicious_url_bar_chart.png', bbox_inches='tight')
plt.show()