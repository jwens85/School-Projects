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

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")  # local version

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

processed_images = []
target_size = (400, 300)  # (width, height)

for image_path in image_file_paths:
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        continue

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Unable to load image: {image_path}")
        continue

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = clahe.apply(grayscale_image)
    enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    detected_faces = face_cascade.detectMultiScale(
        enhanced_image, scaleFactor=1.1, minNeighbors=5
    )

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Eye detection inside face ROI
        face_roi_gray = enhanced_image[y:y + h, x:x + w]
        face_roi_color = original_image[y:y + h, x:x + w]

        detected_eyes = eye_cascade.detectMultiScale(face_roi_gray)

        for (ex, ey, ew, eh) in detected_eyes:
            cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    resized_image = cv2.resize(original_image, target_size)
    processed_images.append(resized_image)

# Combine into a single image grid
if len(processed_images) == 0:
    print("No images to display.")
else:
    # Arrange in 2x2 grid
    row1 = np.hstack(processed_images[:2])
    row2 = np.hstack(processed_images[2:4]) if len(processed_images) > 2 else row1
    final_grid = np.vstack([row1, row2]) if len(processed_images) > 2 else row1

    cv2.imshow("Faces + Eyes - CLAHE + Gaussian", final_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
