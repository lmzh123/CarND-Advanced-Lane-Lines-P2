import cv2
import numpy as np

img = cv2.imread("test_images/straight_lines1.jpg")

pts = np.array([[230,700],[1075,700],[685,450],[595,450]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))

cv2.imshow("Polygon", img)
cv2.waitKey(0)