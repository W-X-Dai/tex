import os
import cv2
import numpy as np

os.makedirs("result/p2", exist_ok=True)

def reduce_gray_levels(img, nLevels=2):
    if nLevels < 2 or nLevels > 256 or (nLevels & (nLevels - 1)):
        raise ValueError("invalid input: nLevels must be a power of 2 and between 2 and 256.")

    delta = 256 // nLevels
    quantized = (img // delta) * delta + delta // 2
    return quantized.astype(np.uint8)

img = cv2.imread("src/Fig1.bmp", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Cannot open Fig1.bmp")

nLevels = int(input("Enter the number of gray levels (should be a power of 2 and between 2 and 256, e.g., 2,4,8,...,256): "))

out = reduce_gray_levels(img, nLevels)

cv2.imwrite(f"result/p2/gray_{nLevels}.bmp", out)
print(f"Result image saved to result/p2/gray_{nLevels}.bmp")

cv2.imshow(f"{nLevels} levels", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
