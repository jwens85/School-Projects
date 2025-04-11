import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image
img = cv2.imread('Image_Noisy.jpg')

# Apply mean filter using a 5x5 kernel
mean_filtered = cv2.blur(img, (5, 5))

# Convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
filtered_rgb = cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2RGB)

# Display original and filtered image side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_rgb)
plt.title('Mean Filtered (5x5)')
plt.axis('off')

plt.tight_layout()
plt.show()
