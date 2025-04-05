import numpy as np

import cv2

img = cv2.imread('Image_Money.jpg')
print(img)

print('Shape of the image: {}'.format(img.shape))

print('Image Height: {}'.format(img.shape[0]))

print('Image Width: {}'.format(img.shape[1]))

print('Image Dimension: {}'.format(img.ndim))
