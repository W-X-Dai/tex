import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def _hist(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,256))
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    return hist

def CDF(img):
    hist, cdf = _hist(img), np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1,256):
        cdf[i] = cdf[i-1] + hist[i]
    return cdf

def hist_equalize(img):
    cdf = CDF(img)
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img_eq = cdf[img]
    return img_eq

if __name__ == "__main__":
    for i in range(1,5):
        img = cv.imread(f'../src/Fig1-{i}.bmp', cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error loading image ../src/Fig1-{i}.bmp")
            continue
        img_eq = hist_equalize(img)
        os.makedirs('result/p1', exist_ok=True)

        # show the origin and equalized image
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title(f'Original 1-{i}')
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title(f'Equalized 1-{i}')
        plt.imshow(img_eq, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'result/p1/hist_equalized_1-{i}.png')

        # show the histograms
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title('Original Histogram')
        hist = _hist(img)
        plt.bar(range(256), hist, width=1.0)
        plt.subplot(1,2,2)
        plt.title('Equalized Histogram')
        hist_eq = _hist(img_eq)
        plt.bar(range(256), hist_eq, width=1.0)
        plt.tight_layout()
        plt.savefig(f'result/p1/hist_equalized_1-{i}_hist.png')