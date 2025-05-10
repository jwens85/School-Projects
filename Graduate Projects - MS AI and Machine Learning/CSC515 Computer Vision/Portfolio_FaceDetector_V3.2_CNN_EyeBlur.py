import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

# --- 1. Initialize detectors ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    thresholds=[0.6, 0.7, 0.7],
    min_face_size=20,
    factor=0.7
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# --- 2. Image paths ---
image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]

processed_images = []
target_size = (400, 300)

for path in image_file_paths:
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"Unable to load image: {path}")
        continue

    # --- 3a. Haar face + Haar eye detection (boxes only) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    faces = face_cascade.detectMultiScale(
        enhanced, scaleFactor=1.1, minNeighbors=5
    )
    for (x, y, w, h) in faces:
        # draw red face box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # detect Haar eyes and draw green boxes
        roi_gray  = enhanced[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20,20)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

    # --- 3b. CNN landmarks → blur eye regions, then draw blue dots ---
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

    if landmarks is not None and boxes is not None:
        for idx, (box, pts) in enumerate(zip(boxes, landmarks)):
            if probs[idx] < 0.75:
                continue

            # landmarks: pts[0]=left eye, pts[1]=right eye
            for eye_pt in (pts[0], pts[1]):
                ex, ey = int(eye_pt[0]), int(eye_pt[1])
                # define square ROI around eye, 20% of face width
                face_w = int(box[2] - box[0])
                eye_rad = int(face_w * 0.20 // 2)
                x1, y1 = max(ex - eye_rad, 0), max(ey - eye_rad, 0)
                x2, y2 = min(ex + eye_rad, img.shape[1]-1), min(ey + eye_rad, img.shape[0]-1)

                # blur that CNN‐eye region with larger kernel
                eye_region = img[y1:y2, x1:x2]
                if eye_region.size:
                    # use a larger blur kernel for stronger obfuscation
                    k = (101, 101)
                    blurred = cv2.GaussianBlur(eye_region, k, 0)
                    img[y1:y2, x1:x2] = blurred

                # draw the CNN eye center (blue dot)
                cv2.circle(img, (ex, ey), 3, (255, 0, 0), -1)

    # --- 4. Resize & collect ---
    processed_images.append(cv2.resize(img, target_size))

# --- 5. Build and display 2×2 grid ---
if processed_images:
    while len(processed_images) < 4:
        processed_images.append(processed_images[-1])
    row1 = np.hstack(processed_images[0:2])
    row2 = np.hstack(processed_images[2:4])
    grid = np.vstack([row1, row2])

    cv2.imshow("Faces (red), Haar Eyes (green), CNN Eyes blurred + blue dots", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images to display.")
