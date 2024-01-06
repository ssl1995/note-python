import numpy as np
import cv2

image = cv2.imread('robot.png')

# 增减对比度
image_1 = cv2.convertScaleAbs(image, alpha=1.5)
image_2 = cv2.convertScaleAbs(image, alpha=0.5)

combined_image = cv2.hconcat([image, image_1, image_2])

cv2.imshow('combined_image', combined_image)
cv2.waitKey(0)
