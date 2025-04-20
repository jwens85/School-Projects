import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

def preprocess_for_stage(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found or cannot be read.")
    return img

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def apply_morph(img, operation, kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(img, operation, kernel)

def apply_dilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(img, kernel, iterations=1)

def apply_erode(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.erode(img, kernel, iterations=1)

def apply_skeleton_overlay(img):
    # EXACT replica of your original code
    _, binarized_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binarized_img = 1 - binarized_img  # Flip foreground/background
    skeletal_lines = skeletonize(binarized_img).astype(np.uint8) * 255

    overlay_base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    orange_highlight = np.zeros_like(overlay_base)
    orange_highlight[skeletal_lines == 255] = [0, 85, 255]  # orange BGR
    composite_skeleton = cv2.addWeighted(overlay_base, 1.0, orange_highlight, 1.0, 0)

    return composite_skeleton

def match_and_draw(original, transformed, label, max_draw_matches=40):
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(original, None)
    kp2, des2 = orb.detectAndCompute(transformed, None)

    if des1 is None or des2 is None:
        match_img = np.hstack((original, transformed))
        return 0, add_text_label(cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR), f"{label}: 0 matches")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        good = [m for i, m in enumerate(good) if mask[i]]

    match_img = cv2.drawMatches(original, kp1, transformed, kp2, good[:max_draw_matches], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good), annotate_matches(match_img, len(good), label)

def annotate_matches(img, count, label):
    annotated = img.copy()
    cv2.rectangle(annotated, (10, 10), (340, 50), (255, 255, 255), -1)
    text = f"{label}: {count} matches"
    cv2.putText(annotated, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return annotated

def add_text_label(img, tag, font_color=(0, 0, 0)):
    cv2.putText(img, tag, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color, 2, cv2.LINE_AA)
    return img

def process_pipeline(input_path):
    original = preprocess_for_stage(input_path)
    clahe = apply_clahe(original)
    dilation = apply_dilate(clahe)
    erosion = apply_erode(clahe)
    opening = apply_morph(clahe, cv2.MORPH_OPEN)
    closing = apply_morph(clahe, cv2.MORPH_CLOSE)
    skeleton_overlay = apply_skeleton_overlay(clahe)

    comparisons = []

    stages = [
        ("CLAHE", clahe),
        ("Dilation", dilation),
        ("Erosion", erosion),
        ("Opening", opening),
        ("Closing", closing),
    ]

    for label, stage_img in stages:
        count, match_img = match_and_draw(original, stage_img, label)
        comparisons.append(match_img)

    # Skeleton handled separately for orange overlay
    skeleton_gray = cv2.cvtColor(skeleton_overlay, cv2.COLOR_BGR2GRAY)
    count, match_img = match_and_draw(original, skeleton_gray, "Skeleton")
    comparisons.append(match_img)

    display_grid(comparisons)

def display_grid(images):
    rows = []
    for i in range(0, len(images), 2):
        row = np.hstack(images[i:i + 2])
        rows.append(row)
    grid = np.vstack(rows)
    cv2.namedWindow("Fingerprint Comparison Grid", cv2.WINDOW_NORMAL)
    cv2.imshow("Fingerprint Comparison Grid", grid)
    cv2.imwrite("Fingerprint_Comparison_Result.png", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_path = r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Fingerprint.png"
    process_pipeline(input_path)
