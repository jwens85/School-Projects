import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN
import warnings

# 0. Suppress facenet_pytorch FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    faces = face_cascade.detectMultiScale(
        enhanced, scaleFactor=1.1, minNeighbors=5
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

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

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

    if landmarks is not None:
        for idx, pts in enumerate(landmarks):
            if probs[idx] < 0.75:
                continue

            left_eye  = tuple(map(int, pts[0]))
            right_eye = tuple(map(int, pts[1]))

            for (fx, fy, fw, fh) in faces:
                if (fx <= left_eye[0] <= fx+fw and fy <= left_eye[1] <= fy+fh and
                    fx <= right_eye[0] <= fx+fw and fy <= right_eye[1] <= fy+fh):
                    cv2.circle(img, left_eye,  3, (255, 0, 0), -1)
                    cv2.circle(img, right_eye, 3, (255, 0, 0), -1)
                    break

    processed_images.append(cv2.resize(img, target_size))

if processed_images:
    while len(processed_images) < 4:
        processed_images.append(processed_images[-1])
    row1 = np.hstack(processed_images[0:2])
    row2 = np.hstack(processed_images[2:4])
    grid = np.vstack([row1, row2])

    cv2.imshow("Faces (red), Haar Eyes (green), CNN Eyes (blue)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images to display.")
