import cv2
import numpy as np

# 讀取灰階圖
img = cv2.imread("src/Fig3.GIF", cv2.IMREAD_GRAYSCALE)

# Step 1: 背景歸一化
background = cv2.GaussianBlur(img, (51, 51), 0)
norm = cv2.divide(img.astype(np.float32), background.astype(np.float32) + 1)
norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Step 2: CLAHE 增強
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(norm)
binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, 20)

cv2.imwrite("mixed_result.png", enhanced)
