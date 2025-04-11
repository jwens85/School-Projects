import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('Image_selfie.jpg')

# Apply Gaussian filter using a 5x5 kernel and sigma = 1.0
gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 1.0)

# Convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
filtered_rgb = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB)

# Display original and Gaussian-filtered image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_rgb)
plt.title('Gaussian Filtered (5x5, σ=1.0)')
plt.axis('off')

plt.tight_layout()
plt.show()
