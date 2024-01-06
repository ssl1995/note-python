'''
特征值：与特征向量相对应的标量称为特征值，表示在特定的特征向量方向上，变换的缩放比例，反应了这个变化增强、减弱的程度
特征向量：如果一个非零向量在一个线性变换行的效果仅仅是被伸缩，而方向不变，这个向量称为特征向量，类似那根被拉伸后方向不变的箭头，表明变换的主要方向
'''
import numpy as np

A = np.array([
    [4, 1],
    [2, 3]])

# 求A的特征值和特征向量
eigenvalues, eigenvators = np.linalg.eig(A)

print(f"特征值：{eigenvalues}")  # 5,2 = 伸缩的程度
print(f"特征向量：{eigenvators}")  # 伸缩的方向
