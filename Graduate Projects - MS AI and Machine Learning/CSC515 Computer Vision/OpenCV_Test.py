import cv2

# Load an image (Replace 'your_image.jpg' with an actual image file path)
image = cv2.imread(r"C:\Users\jwens\Desktop\CSUGlobal\CSC510\CSC510 Portfolio Project\Test Image 1.png")

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    # Display the image
    cv2.imshow('OpenCV Image', image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
