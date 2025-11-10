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

# Input threshold
thresh = input("Enter threshold value (0-255): ")
try:
    thresh = int(thresh)
    if not (0 <= thresh <= 255):
        raise ValueError
except ValueError:
    raise ValueError("Invalid threshold value. Please enter an integer between 0 and 255.")

# Binarize image based on threshold
img = np.zeros_like(img_ori, dtype=np.uint8)
img[img_ori >= thresh] = 255

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_ori, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Processed Image')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('result/p2/2a.png', dpi=300)