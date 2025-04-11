import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in grayscale
img = cv2.imread('Image_Noisy.jpg', cv2.IMREAD_GRAYSCALE)

# Optional: smooth first with Gaussian to reduce noise
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)

# Apply Laplacian
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# Convert to displayable format
laplacian_display = cv2.convertScaleAbs(laplacian)

# Show original and Laplacian edge-detected image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(laplacian_display, cmap='gray')
plt.title('Laplacian Filtered')
plt.axis('off')

plt.tight_layout()
plt.show()
