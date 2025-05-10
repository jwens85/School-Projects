import cv2
import os
import numpy as np

image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]

frontal_face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for image_index, image_path in enumerate(image_file_paths):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        continue

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Unable to load image: {image_path}")
        continue

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = clahe.apply(grayscale_image)
    enhanced_image = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=55)

    laplacian_edges = cv2.Laplacian(enhanced_image, cv2.CV_8U, ksize=3)

    # Blend the Laplacian edge mask with the preprocessed image
    hybrid_image = cv2.addWeighted(enhanced_image, 0.7, laplacian_edges, 0.3, 0)

    detected_faces = frontal_face_classifier.detectMultiScale(
        hybrid_image,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    window_title = f"Image {image_index + 1} - Hybrid Laplacian Detection"
    cv2.imshow(window_title, original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
