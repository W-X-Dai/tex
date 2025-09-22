import cv2
import os
import utils

os.makedirs("result/p4", exist_ok=True)

img = cv2.imread("src/Fig3.GIF", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Cannot open Fig3.GIF")

binary = cv2.adaptiveThreshold(img, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,
                               29, 10)
cv2.imwrite("result/p4/text_recovered1.bmp", binary)



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_eq = clahe.apply(img)

binary = cv2.adaptiveThreshold(img_eq, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,
                               35, 10)

cv2.imwrite("result/p4/text_recovered2.bmp", binary)
print("Saved result to result/p4/text_recovered2.bmp")
