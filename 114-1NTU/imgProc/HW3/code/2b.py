import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

os.makedirs('result/p2', exist_ok=True)

# Read image
img_ori = cv2.imread('../src/Fig2.gif', 0)
if img_ori is None:
    raise ValueError("Image not found or unable to load.")

thresh = 128
img = np.zeros_like(img_ori, dtype=np.uint8)
img[img_ori >= thresh] = 255

# Erosion
eroded = np.zeros_like(img, dtype=np.uint8)
H, W = img.shape
kernel = np.ones((3, 3), np.uint8)
for i in range(1, H-1):
    for j in range(1, W-1):
        region = img[i-1:i+2, j-1:j+2]
        if np.all(region == 255):
            eroded[i, j] = 255
            
edge = img - eroded

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_ori, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Detected Edge')
plt.imshow(edge, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('result/p2/2b.png', dpi=300)