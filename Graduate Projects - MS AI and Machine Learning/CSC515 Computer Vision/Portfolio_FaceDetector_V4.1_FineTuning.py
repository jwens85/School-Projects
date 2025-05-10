import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# ------------------------
# Helpers
# ------------------------
def iou(boxA, boxB):
    # compute intersection over union between two boxes (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def blur_roi(roi, ksize=(31,31)):
    return cv2.GaussianBlur(roi, ksize, 0) if roi.size else roi

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
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
base_radius   = 10  # half‑size of CNN eye blur region

for path in image_file_paths:
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"Unable to load image: {path}")
        continue

    haar_eye_regions = []

    # --- Haar face (red) + Haar eyes (green), blur original color ---
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    faces = face_cascade.detectMultiScale(enhanced, scaleFactor=1.1, minNeighbors=5)
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)

        # detect Haar eyes within this face ROI
        raw = eye_cascade.detectMultiScale(
            enhanced[fy:fy + fh, fx:fx + fw],
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        # convert to global coords
        boxes = [
            (fx + ex, fy + ey, fx + ex + ew, fy + ey + eh)
            for (ex, ey, ew, eh) in raw
        ]
        # sort by area ascending (smallest first)
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        # non-max suppression + limit to two smallest
        kept = []
        for b in boxes:
            if not any(iou(b, k) > 0.3 for k in kept):
                kept.append(b)
            if len(kept) == 2:
                break

        for (x1, y1, x2, y2) in kept:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            haar_eye_regions.append((x1, y1, x2, y2))
            roi = img[y1:y2, x1:x2]
            img[y1:y2, x1:x2] = blur_roi(roi)

    # --- CNN face detection (blue) + eye landmarks (no blue dots) ---
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

    if landmarks is not None:
        for i in accepted_idxs:
            pts = landmarks[i]
            # blur around each eye without drawing dots
            for eye_pt in (pts[0], pts[1]):
                ex, ey = map(int, eye_pt)
                if any(hx1 <= ex <= hx2 and hy1 <= ey <= hy2
                       for (hx1, hy1, hx2, hy2) in haar_eye_regions):
                    continue
                r  = base_radius
                y1 = max(0, ey - r)
                y2 = min(img.shape[0], ey + r)
                x1 = max(0, ex - r)
                x2 = min(img.shape[1], ex + r)
                roi = img[y1:y2, x1:x2]
                img[y1:y2, x1:x2] = blur_roi(roi)

    processed_images.append(cv2.resize(img, target_size))

# display in 2×2 grid
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
