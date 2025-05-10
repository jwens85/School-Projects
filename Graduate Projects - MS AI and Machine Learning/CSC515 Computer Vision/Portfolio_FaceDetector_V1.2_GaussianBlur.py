#V1.2 with Gaussian blurring
import cv2
import os

# List of image file paths
image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]

# Load Haar cascade for face detection
frontal_face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# CLAHE configuration
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

    # Apply Gaussian blur after CLAHE
    enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    detected_faces = frontal_face_classifier.detectMultiScale(
        enhanced_image, scaleFactor=1.1, minNeighbors=5
    )

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    window_title = f"Image {image_index + 1} - Gaussian CLAHE"
    cv2.imshow(window_title, original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
