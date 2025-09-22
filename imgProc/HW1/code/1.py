import cv2
import numpy as np

img = cv2.imread("src/Fig1.bmp", cv2.IMREAD_GRAYSCALE)
if img is None:
	print("Error: Unable to load image.")
	exit()

top = (150, 150)
h = int((3**0.5 / 2) * 200) 
left = (150 - 100, 150 + h)
right = (150 + 100, 150 + h)

pts = np.array([top, left, right], np.int32)

cv2.fillPoly(img, [pts], 255)

cv2.imwrite("result/1.bmp", img)
cv2.imshow("Triangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
