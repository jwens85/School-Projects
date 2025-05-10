import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# 1) Initialize MTCNN for CNN face detection + landmarks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    thresholds=[0.6, 0.7, 0.7],
    min_face_size=20,
    factor=0.7
)

# 2) Haar cascades for face proposals and eye detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# 3) CLAHE for contrast enhancement before Haar
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# simple IoU helper
def iou(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    xi1, yi1 = max(xA1, xB1), max(yA1, yB1)
    xi2, yi2 = min(xA2, xB2), min(yA2, yB2)
    interW = max(0, xi2 - xi1)
    interH = max(0, yi2 - yi1)
    interA = interW * interH
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union = areaA + areaB - interA
    return interA / union if union > 0 else 0

# List of image file paths (adjust these to your actual files)
image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]

processed      = []
target_size    = (400, 300)
cnn_conf_thresh = 0.75
iou_thresh      = 0.5

for path in image_file_paths:
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"Unable to load image: {path}")
        continue

    # Prepare grayscale + enhanced image for Haar
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # --- Haar face proposals + green eye boxes ---
    haar_faces = []
    faces = face_cascade.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        bbox = (x, y, x + w, y + h)

        # detect eyes inside this face ROI
        roi_gray = enhanced[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        eye_rects = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes]
        haar_faces.append((bbox, eye_rects))

        # draw red face box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # draw green eye boxes
        for (ex, ey, ew, eh) in eye_rects:
            cv2.rectangle(
                img[y:y+h, x:x+w],
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

    # --- CNN (MTCNN) face proposals + eye landmarks/dots ---
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

    if boxes is not None and landmarks is not None:
        for i, (box, prob, pts) in enumerate(zip(boxes, probs, landmarks)):
            if prob < cnn_conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box)
            # draw blue face box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # dynamic dot radius
            face_w = x2 - x1
            dot_r  = max(2, int(face_w * 0.03))

            # attempt MTCNN landmarks
            use_pts = False
            if pts is not None:
                le = tuple(map(int, pts[0]))
                re = tuple(map(int, pts[1]))
                if (x1 <= le[0] <= x2 and y1 <= le[1] <= y2 and
                    x1 <= re[0] <= x2 and y1 <= re[1] <= y2):
                    use_pts = True

            if use_pts:
                # draw blue dots at MTCNN landmarks
                cv2.circle(img, le, dot_r, (255, 0, 0), -1)
                cv2.circle(img, re, dot_r, (255, 0, 0), -1)
            else:
                # fallback: match to a Haar face
                matched_eyes = None
                for (hf_bbox, hf_eyes) in haar_faces:
                    if iou(hf_bbox, (x1, y1, x2, y2)) >= iou_thresh and len(hf_eyes) >= 2:
                        matched_eyes = hf_eyes
                        break

                if matched_eyes:
                    # use first two Haar eyes
                    e0, e1 = matched_eyes[0], matched_eyes[1]
                    c0 = (x1 + e0[0] + e0[2]//2, y1 + e0[1] + e0[3]//2)
                    c1 = (x1 + e1[0] + e1[2]//2, y1 + e1[1] + e1[3]//2)
                    cv2.circle(img, c0, dot_r, (255, 0, 0), -1)
                    cv2.circle(img, c1, dot_r, (255, 0, 0), -1)
                else:
                    # final fallback: Haar-eye on CNN ROI
                    roi_gray = enhanced[y1:y2, x1:x2]
                    fb_eyes  = eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(20, 20)
                    )
                    fb_eyes = sorted(fb_eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                    if len(fb_eyes) < 2:
                        continue
                    fb_eyes.sort(key=lambda e: e[0])
                    ex0, ey0, ew0, eh0 = fb_eyes[0]
                    ex1, ey1, ew1, eh1 = fb_eyes[1]
                    c0 = (x1 + ex0 + ew0//2, y1 + ey0 + eh0//2)
                    c1 = (x1 + ex1 + ew1//2, y1 + ey1 + eh1//2)
                    cv2.circle(img, c0, dot_r, (255, 0, 0), -1)
                    cv2.circle(img, c1, dot_r, (255, 0, 0), -1)

    # resize for uniform display
    processed.append(cv2.resize(img, target_size))

# final 2×2 grid display
if processed:
    while len(processed) < 4:
        processed.append(processed[-1])
    top_row    = np.hstack(processed[0:2])
    bottom_row = np.hstack(processed[2:4])
    grid       = np.vstack([top_row, bottom_row])
    cv2.imshow("Haar faces:red  Haar eyes:green  CNN faces:blue  CNN eyes:blue dots", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images to display.")
