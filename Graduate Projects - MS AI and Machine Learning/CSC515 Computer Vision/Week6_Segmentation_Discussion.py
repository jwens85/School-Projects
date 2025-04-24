import cv2
import numpy as np

image_path = r'C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Segmentation_Discussion.png'
img_original = cv2.imread(image_path)
gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 3)

edges = cv2.Canny(gray, 100, 200)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges_for_contours = cv2.Canny(blurred, 30, 100)

contours, _ = cv2.findContours(edges_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filled_mask = np.zeros_like(gray)
min_area = 1000

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        cv2.drawContours(filled_mask, [cnt], -1, 255, cv2.FILLED)

cv2.imshow('a) Original', img_original)
cv2.imshow('b) Thresholding (Adaptive)', thresh_adaptive)
cv2.imshow('c) Canny Edges', edges)
cv2.imshow('d) Region-Based (Contour Area Filter)', filled_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
