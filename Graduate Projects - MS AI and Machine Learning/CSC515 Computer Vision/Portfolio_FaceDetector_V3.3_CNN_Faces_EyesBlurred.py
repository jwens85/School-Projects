import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# initialize MTCNN for reliable CNN detections + landmarks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    thresholds=[0.6, 0.7, 0.7],
    min_face_size=20,
    factor=0.7
)

# Haar cascade for initial face proposals (used only if confirmed by CNN)
haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# simple IoU helper
def iou(boxA, boxB):
    xA1,yA1,xA2,yA2 = boxA
    xB1,yB1,xB2,yB2 = boxB
    xi1, yi1 = max(xA1,xB1), max(yA1,yB1)
    xi2, yi2 = min(xA2,xB2), min(yA2,yB2)
    interW = max(0, xi2-xi1)
    interH = max(0, yi2-yi1)
    interA = interW * interH
    areaA = (xA2-xA1)*(yA2-yA1)
    areaB = (xB2-xB1)*(yB2-yB1)
    union = areaA + areaB - interA
    return interA/union if union>0 else 0

# your four image paths
image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]

processed, target_size = [], (400, 300)
cnn_conf_thresh, iou_thresh = 0.75, 0.5

for path in image_file_paths:
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"Unable to load image: {path}")
        continue

    # preprocess for Haar
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enh  = clahe.apply(gray)
    enh  = cv2.GaussianBlur(enh, (5,5), 0)

    # 1) Haar face proposals
    haar_boxes = []
    for (x,y,w,h) in haar.detectMultiScale(enh, 1.1, 5):
        haar_boxes.append((x, y, x+w, y+h))

    # 2) CNN face proposals + landmarks
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

    # prepare matched sets
    matched = []
    if boxes is not None and landmarks is not None:
        for i, (box, prob, pts) in enumerate(zip(boxes, probs, landmarks)):
            if prob < cnn_conf_thresh:
                continue
            x1,y1,x2,y2 = map(int, box)
            # check overlap with any Haar box
            for hb in haar_boxes:
                if iou((x1,y1,x2,y2), hb) >= iou_thresh:
                    matched.append((hb, pts))
                    break

    # draw matched Haar boxes + two blue dots from CNN landmarks
    for (x1,y1,x2,y2), pts in matched:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)              # Haar = red
        le = tuple(map(int, pts[0])); re = tuple(map(int, pts[1]))
        cv2.circle(img, le,  3, (255,0,0), -1)                         # blue eye dots
        cv2.circle(img, re,  3, (255,0,0), -1)

    # draw remaining CNN-only faces
    if boxes is not None and landmarks is not None:
        for box, prob, pts in zip(boxes, probs, landmarks):
            if prob < cnn_conf_thresh:
                continue
            x1,y1,x2,y2 = map(int, box)
            # skip if already drawn via Haar match
            if any(iou((x1,y1,x2,y2), hb) >= iou_thresh for hb,_ in matched):
                continue
            # draw CNN-only face box in blue
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            le = tuple(map(int, pts[0])); re = tuple(map(int, pts[1]))
            cv2.circle(img, le,  3, (255,0,0), -1)
            cv2.circle(img, re,  3, (255,0,0), -1)

    processed.append(cv2.resize(img, target_size))

# final display in 2×2 grid
if processed:
    while len(processed) < 4:
        processed.append(processed[-1])
    top = np.hstack(processed[0:2])
    bot = np.hstack(processed[2:4])
    grid= np.vstack([top, bot])
    cv2.imshow("Haar (red)+CNN landmarks; CNN-only (blue)+eyes", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images to display.")
