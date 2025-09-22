import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

"""This is used to be a bilinear interpolation function I've finished in the past, but I changed to nearest neighbor interpolation for simplicity."""
def bi_affine(img, A):
    img_out = np.zeros_like(img)
    inv = np.linalg.inv(A)

    h, w = img.shape
    for i in range(h):          # row
        for j in range(w):      # col
            origin_T = np.array([[j], [i], [1]])
            inv_T = inv @ origin_T

            c, r = inv_T[0].item(), inv_T[1].item()

            # clamp
            r = min(max(r, 0), h-1)
            c = min(max(c, 0), w-1)

            # nearest neighbor
            r_nn, c_nn = int(round(r)), int(round(c))
            img_out[i, j] = img[r_nn, c_nn]

    return img_out

def get_rotation_matrix(h, w, angle_deg):
    cx, cy = w/2, h/2
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    A = np.array([
        [ cos_t, sin_t, (1-cos_t)*cx-sin_t*cy],
        [-sin_t, cos_t, (1-cos_t)*cy+sin_t*cx],
        [     0,     0,                     1]
    ], dtype=np.float32)

    return A

if __name__ == '__main__':
    os.makedirs("result/p3", exist_ok=True)

    img = cv2.imread('src/Fig2.bmp', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Cannot find Fig2.bmp")
    
    h, w = img.shape

    angles = [15, 30, 45, 60, 90]
    results = [img]

    for angle in angles:
        A = get_rotation_matrix(h, w, -angle)
        rotated = bi_affine(img, A)

        cv2.imwrite(f"result/p3/rot_{angle}.bmp", rotated)
        results.append(rotated)

    print("Result images saved to result/p3/")

    plt.figure(figsize=(12, 8))
    titles = ["Original"] + [f"{a}° CW" for a in angles]

    for idx, (title, im) in enumerate(zip(titles, results), 1):
        plt.subplot(2, 3, idx)
        plt.imshow(im, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("result/p3/combined_grid.png")
    plt.show()