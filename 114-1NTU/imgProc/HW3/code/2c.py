import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from math import cos, sin, pi

os.makedirs('result/p2', exist_ok=True)

# Read image
img_ori = cv2.imread('../src/Fig2.gif', 0)
if img_ori is None:
    raise ValueError("Image not found or unable to load.")

# Get the eroded image
thresh = 128
img = np.zeros_like(img_ori, dtype=np.uint8)
img[img_ori >= thresh] = 255

eroded = np.zeros_like(img, dtype=np.uint8)
H, W = img.shape
kernel = np.ones((3, 3), np.uint8)
for i in range(1, H-1):
    for j in range(1, W-1):
        region = img[i-1:i+2, j-1:j+2]
        if np.all(region == 255):
            eroded[i, j] = 255
            
edge = img - eroded

# Set the Hough Transform parameters
min_r, max_r, r_step, theta_res = 10, 30, 2, 60
acc = np.zeros((H, W, (max_r - min_r)//r_step), dtype=np.uint16)

# Accumulate votes in the Hough space
ys, xs = np.nonzero(edge)
for x, y in zip(xs, ys):
    for r_idx, r in enumerate(range(min_r, max_r, r_step)):
        for t in range(theta_res):
            theta = 2 * pi * t / theta_res
            a = int(x - r * cos(theta))
            b = int(y - r * sin(theta))
            if 0 <= a < W and 0 <= b < H:
                acc[b, a, r_idx] += 1

b, a, r_idx = np.unravel_index(np.argmax(acc), acc.shape)
r = min_r + r_idx * r_step

print(f"Detected circle: center=({a},{b}), radius={r}")

result = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2BGR)
cv2.circle(result, (int(a), int(b)), int(r), (0, 255, 0), 2)
cv2.circle(result, (int(a), int(b)), 2, (0, 0, 255), 3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_ori, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Detected Circle')
plt.imshow(result)
plt.axis('off')
plt.tight_layout()
plt.savefig('result/p2/2c.png', dpi=300)