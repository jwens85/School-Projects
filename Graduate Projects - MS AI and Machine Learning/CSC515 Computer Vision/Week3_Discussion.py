import cv2

# Load the image
img = cv2.imread('Image_Faces.png')

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a CLAHE object with specific parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Apply CLAHE to enhance local contrast
enhanced_img = clahe.apply(gray_img)

# Display the results
cv2.imshow('Original Image', gray_img)
cv2.imshow('CLAHE Enhanced Image', enhanced_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
