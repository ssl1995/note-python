import matplotlib.pyplot as plt
import numpy as np


labels = ['RN50', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336', 'ViT-H/14']
men_means = [26.6, 32.0, 36.8, 37.2, 48.6]
women_means = [57.0, 65.4, 68.9, 71.7, 74]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='CLIP')
rects2 = ax.bar(x + width/2, women_means, width, label='CN_CLIP')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Accuracy')
ax.set_title('Compare')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

