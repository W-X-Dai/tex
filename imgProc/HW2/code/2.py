import os
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img = cv.imread('../src/Fig2-1.bmp', cv.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found or could not be opened.")
os.makedirs('result/p2', exist_ok=True)

sizes = [1, 3, 5, 9, 15, 35]
for m in sizes:
    smooth = cv.blur(img, (m, m))
    plt.figure()
    plt.title(f"Averaging Filter m={m}")
    plt.imshow(smooth, cmap='gray')
    plt.axis('off')
    plt.savefig(f'result/p2/averaging_filter_m_{m}.png')