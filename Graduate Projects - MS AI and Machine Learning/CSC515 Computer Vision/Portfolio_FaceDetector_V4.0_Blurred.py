import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# ------------------------
# 1) Configure MTCNN
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    thresholds=[0.3, 0.3, 0.4],
    min_face_size=15,
    factor=0.7
)

# ------------------------
# 2) Haar cascades & CLAHE
# ------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade  = cv2.CascadeClassifier("haarcascade_eye.xml")
clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ------------------------
# 3) Image paths & sizing
# ------------------------
image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]
processed_images = []
target_size = (400, 300)

# parameters for CNN‑eye blur
base_radius   = 10   # half‑width/height of the CNN eye blur region

for path in image_file_paths:
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"Unable to load image: {path}")
        continue

    # track Haar‑detected eye regions to suppress CNN blur later
    haar_eye_regions = []

    # --- Haar face (red) + Haar eyes (green), blur original color ---
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    faces = face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=5)
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        roi_gray  = enhanced[fy:fy + fh, fx:fx + fw]
        roi_color = img[fy:fy + fh, fx:fx + fw]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            # draw green box
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # compute a slightly smaller region inside the eye box (20% margin)
            mx = int(ew * 0.2)
            my = int(eh * 0.2)
            x1 = fx + ex + mx
            y1 = fy + ey + my
            x2 = fx + ex + ew - mx
            y2 = fy + ey + eh - my

            # record this Haar eye region
            haar_eye_regions.append((x1, y1, x2, y2))

            # blur the original color in that region
            roi_eye = img[y1:y2, x1:x2]
            if roi_eye.size != 0:
                blurred = cv2.GaussianBlur(roi_eye, (31, 31), 0)
                img[y1:y2, x1:x2] = blurred

    # --- CNN face detection (blue) + eye landmarks ---
    rgb                     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

    accepted_idxs = []
    if boxes is not None and probs is not None:
        for i, (box, p) in enumerate(zip(boxes, probs)):
            if p < 0.85:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            accepted_idxs.append(i)

    # blur around each CNN landmark eye, unless covered by Haar blur
    if landmarks is not None:
        for i in accepted_idxs:
            pts = landmarks[i]
            for eye_pt in (pts[0], pts[1]):
                ex, ey = map(int, eye_pt)

                # draw the blue dot in all cases
                cv2.circle(img, (ex, ey), 3, (255, 0, 0), -1)

                # skip blur if inside any Haar‑blur region
                if any(hx1 <= ex <= hx2 and hy1 <= ey <= hy2
                       for (hx1, hy1, hx2, hy2) in haar_eye_regions):
                    continue

                # define blur square (10 px half‑size)
                r = base_radius
                y1 = max(0, ey - r)
                y2 = min(img.shape[0], ey + r)
                x1 = max(0, ex - r)
                x2 = min(img.shape[1], ex + r)

                if y2 <= y1 or x2 <= x1:
                    continue

                # blur the original color in that region
                roi_eye = img[y1:y2, x1:x2]
                if roi_eye.size != 0:
                    blurred = cv2.GaussianBlur(roi_eye, (31, 31), 0)
                    img[y1:y2, x1:x2] = blurred

    # resize and collect
    processed_images.append(cv2.resize(img, target_size))

# --- display 2×2 grid ---
if processed_images:
    while len(processed_images) < 4:
        processed_images.append(processed_images[-1])
    row1 = np.hstack(processed_images[0:2])
    row2 = np.hstack(processed_images[2:4])
    grid = np.vstack([row1, row2])
    cv2.imshow("Faces & Blurred Eyes", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images to display.")
