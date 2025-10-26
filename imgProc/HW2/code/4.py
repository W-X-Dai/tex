import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a = cv.imread('../src/Fig4-1.bmp', cv.IMREAD_GRAYSCALE)
if a is None:
    raise ValueError("Image not found or could not be opened.")
a = a.astype(np.float32)
os.makedirs('result/p4', exist_ok=True)

b = cv.Laplacian(a, cv.CV_32F, ksize=3)

c = cv.add(a, b)

gx = cv.Sobel(a, cv.CV_32F, 1, 0, ksize=3)
gy = cv.Sobel(a, cv.CV_32F, 0, 1, ksize=3)
d = cv.magnitude(gx, gy)

e = cv.blur(d, (5,5))

f = cv.multiply(c, e, scale=1/255.0)

g = cv.add(a, f)

gamma = 0.5
g_norm = cv.normalize(g, None, 0, 1, cv.NORM_MINMAX)
power = np.power(g_norm, gamma)
h = cv.normalize(power, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

titles = ['(a) Original', '(b) Laplacian', '(c) a+b', '(d) Sobel',
          '(e) Smoothed', '(f) Product', '(g) Sum', '(h) Power-law']
images = [a, b, c, d, e, f, g, h]

plt.figure(figsize=(16,8))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig('result/p4/enhancement_steps.png')
