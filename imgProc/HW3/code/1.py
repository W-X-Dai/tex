import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

os.makedirs('result/p1', exist_ok=True)

# Read image
img_ori = cv2.imread('../src/Fig1.tif', 0)
if img_ori is None:
    raise ValueError("Image not found or unable to load.")

img = (img_ori < 128).astype(np.uint8) * 255  # Binarize
print(img.shape)

H, W = img.shape
dirs = [(-1, -1), (-1,  0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1,  0), ( 1, 1)]
visited = np.zeros((H, W), dtype=bool)
label_map = np.zeros((H, W), dtype=int)
areas = []
label = 0

# BFS for connected components
for i in range(H):
    for j in range(W):
        if img[i, j] and visited[i, j] == 0:
            label += 1
            area = 0
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                if visited[x, y]:
                    continue
                visited[x, y] = 1
                label_map[x, y] = label
                area += 1
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if img[nx, ny] and not visited[nx, ny]:
                            stack.append((nx, ny))
            areas.append(area)

print(f'Total connected components: {label}')

# Find the threshold to remove small components
areas_sorted = sorted(areas, reverse=True)
max_dist = 0
max_thresh = 0

for i in range(label-1):
    dist = abs(areas_sorted[i] - areas_sorted[i+1])
    if dist > max_dist:
        max_dist = dist
        max_thresh = (areas_sorted[i] + areas_sorted[i+1]) / 2

removed = np.zeros(label, dtype=bool)
for i in range(label):
    removed[i] = areas[i] < max_thresh

# Create new image with small components removed
img_new = img_ori.copy()
pre_remove = img_ori.copy()

# generate a binary mask for removed components
mask = (label_map > 0) & removed[label_map - 1]
img_new[mask] = 255
pre_remove[mask] = 128


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(pre_remove, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Processed Image')
plt.imshow(img_new, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('result/p1/1.png', dpi=300)