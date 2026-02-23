import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img = cv.imread('../src/Fig3-1.bmp', cv.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found or could not be opened.")
os.makedirs('result/p3', exist_ok=True)


# Roberts mask
Gx = np.array([[1, 0],
               [0, -1]], dtype=np.float32)

Gy = np.array([[0, 1],
               [-1, 0]], dtype=np.float32)

# Convolve with Roberts masks
grad_x = cv.filter2D(img, cv.CV_32F, Gx)
grad_y = cv.filter2D(img, cv.CV_32F, Gy)

# Compute gradient magnitude
grad = cv.magnitude(grad_x, grad_y)
grad = cv.normalize(grad, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
grad = grad.astype(np.uint8)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Roberts Cross Gradient')
plt.imshow(grad, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('result/p3/roberts_cross_gradient.png')