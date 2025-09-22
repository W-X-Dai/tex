import os
import cv2
import numpy as np

os.makedirs("result/p1", exist_ok=True)

img = cv2.imread("src/Fig1.bmp", cv2.IMREAD_GRAYSCALE)
if img is None:
	print("Error: Unable to load image.")
	exit()

"""To calculate the corner coordinates of an equilateral triangle."""
top = (150, 150)
h = int((3**0.5 / 2) * 200) 
left = (150 - 100, 150 + h)
right = (150 + 100, 150 + h)

pts = np.array([top, left, right], np.int32)

cv2.fillPoly(img, [pts], 255)

cv2.imwrite("result/p1/output.bmp", img)
print("Result image saved to result/p1/output.bmp")

cv2.imshow("Triangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
