import cv2
import numpy as np
import os


def preprocess_fingerprint(image_path, use_morph=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    if use_morph:
        # Tuned kernel size and iteration to avoid over-smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)

    return enhanced


def annotate_matches(image, match_count, label):
    img = image.copy()
    cv2.rectangle(img, (10, 10), (300, 45), (255, 255, 255), -1)
    text = f"{label}: {match_count} matches"
    cv2.putText(img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return img


def match_fingerprints(img1_path, img2_path, use_morph, label, max_draw_matches=30):
    img1_enh = preprocess_fingerprint(img1_path, use_morph)
    img2_enh = preprocess_fingerprint(img2_path, use_morph)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img1_enh, None)
    kp2, des2 = orb.detectAndCompute(img2_enh, None)

    if des1 is None or des2 is None:
        return 0, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        good_matches = [m for i, m in enumerate(good_matches) if mask[i]]

    match_img = cv2.drawMatches(img1_enh, kp1, img2_enh, kp2,
                                good_matches[:max_draw_matches], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_img = annotate_matches(match_img, len(good_matches), label)
    return len(good_matches), match_img


def compare_and_display(img1_path, img2_path, threshold=15):
    count_no_morph, img_no_morph = match_fingerprints(img1_path, img2_path, use_morph=False, label="No Morph")
    count_with_morph, img_with_morph = match_fingerprints(img1_path, img2_path, use_morph=True, label="With Morph")

    print(f"\n--- Comparing {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} ---")
    print(f"[NO Morph]    Matches: {count_no_morph} => {'MATCHED' if count_no_morph > threshold else 'UNMATCHED'}")
    print(f"[WITH Morph]  Matches: {count_with_morph} => {'MATCHED' if count_with_morph > threshold else 'UNMATCHED'}")

    if img_no_morph is not None:
        cv2.imshow("NO Morphological Closing", img_no_morph)

    if img_with_morph is not None:
        cv2.imshow("WITH Morphological Closing", img_with_morph)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img1 = r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\dataset_folder\data1\101\101_1.tif"
    img2 = r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\dataset_folder\data1\101\101_2.tif"

    if not os.path.exists(img1) or not os.path.exists(img2):
        raise FileNotFoundError("Check that your test image paths are valid.")

    compare_and_display(img1, img2)
