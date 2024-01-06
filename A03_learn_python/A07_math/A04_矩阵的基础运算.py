import numpy as np
import cv2


A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = 3

# 加法
add = A + B
# 乘法
multi = C * A

print(f"add:{add}")
print(f"multi:{multi}")