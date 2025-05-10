import cv2
import os

# List of image file paths
image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]

# Load Haar cascade classifier for face detection
frontal_face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Tuned bilateral filter parameters
bilateral_d = 9
bilateral_sigmaColor = 75
bilateral_sigmaSpace = 55

# Process each image
for image_index, image_path in enumerate(image_file_paths):
    if not os.path.exists(image_path):
        print(f"[!] File not found: {image_path}")
        continue

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"[!] Failed to load image: {image_path}")
        continue

    # Preprocessing
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = clahe.apply(grayscale_image)
    enhanced_image = cv2.bilateralFilter(
        enhanced_image,
        d=bilateral_d,
        sigmaColor=bilateral_sigmaColor,
        sigmaSpace=bilateral_sigmaSpace
    )

    # Detect faces
    detected_faces = frontal_face_classifier.detectMultiScale(
        enhanced_image,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the result
    window_title = f"Image {image_index + 1} - Tuned Detection"
    cv2.imshow(window_title, original_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
