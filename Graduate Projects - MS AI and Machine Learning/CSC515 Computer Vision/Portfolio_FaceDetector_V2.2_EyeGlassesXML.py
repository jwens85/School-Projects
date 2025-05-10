import cv2
import os
import numpy as np

# List of image file paths
image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade_a = cv2.CascadeClassifier("haarcascade_eye.xml")
eye_cascade_b = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# CLAHE config
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

target_size = (400, 300)
processed_images = []

for image_path in image_file_paths:
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    faces = face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        roi_gray = enhanced[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Detector A: GREEN
        eyes_a = eye_cascade_a.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes_a:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detector B: YELLOW
        eyes_b = eye_cascade_b.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes_b:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

    resized = cv2.resize(image, target_size)
    processed_images.append(resized)

# Display stacked results
if processed_images:
    row1 = np.hstack(processed_images[:2])
    row2 = np.hstack(processed_images[2:4]) if len(processed_images) > 2 else row1
    grid = np.vstack([row1, row2]) if len(processed_images) > 2 else row1

    cv2.imshow("Eye A/B Comparison - Green=eye.xml, Yellow=eye_tree_eyeglasses", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images processed.")
