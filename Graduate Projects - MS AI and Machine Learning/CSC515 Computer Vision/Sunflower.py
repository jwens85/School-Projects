import cv2

img = cv2.imread('sunflower.jpg')

cv2.imwrite('sunflower_copy.png', img)

cv2.imshow('sunflower_window', img)

cv2.waitKey(0)

cv2.destroyAllWindows()