import cv2
import os

os.makedirs("result/p4", exist_ok=True)

img = cv2.imread("src/Fig3.GIF", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Cannot open Fig3.GIF")


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_eq = clahe.apply(img)

for b_size in range(11, 55, 2):
    for C in range(10, 11, 1):
        binary = cv2.adaptiveThreshold(img_eq, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       b_size, C)

        # img_out = cv2.medianBlur(img_eq, b_size)
        cv2.imwrite(f"result/p4/text_recovered_b{b_size}_C{C}.bmp", binary)

# binary = cv2.adaptiveThreshold(img_eq, 255,
#                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY,
#                                11, 15)
# cv2.imwrite("result/p4/text_recovered0.bmp", binary)

print("Saved result to result/p4/text_recovered.bmp")
