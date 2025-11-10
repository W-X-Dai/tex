import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

os.makedirs('result/p1', exist_ok=True)

# Read image
img = cv2.imread('../src/Fig1.tif', 0)
if img is None:
    raise ValueError("Image not found or unable to load.")

img = (img < 128).astype(np.uint8) * 255  # Binarize
print(img.shape)

H, W = img.shape
dirs = [(-1, -1), (-1,  0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1,  0), ( 1, 1)]
visited = np.zeros((H, W), dtype=bool)
label_map = np.zeros((H, W), dtype=int)
areas = []
label = 0

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

