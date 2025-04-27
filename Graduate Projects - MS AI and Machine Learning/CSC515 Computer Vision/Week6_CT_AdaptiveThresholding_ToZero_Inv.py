import cv2
import numpy as np

# --- Configuration ---------------------------------------------------------

# Paths to your three input images
image_paths = {
    "Indoor":  r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Inside.jpg",
    "Outdoor": r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Outside.jpg",
    "Closeup": r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Closeup.jpg"
}

# CLAHE (Contrast-Limited Adaptive Histogram Equalization)
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)

# Bilateral filter (edge-preserving smoothing)
BILAT_DIAM   = 9
BILAT_SIGMAC = 75
BILAT_SIGMAS = 75

# Adaptive threshold parameters
ADAPT_METHOD = cv2.ADAPTIVE_THRESH_MEAN_C
THRESH_TYPE  = cv2.THRESH_BINARY_INV
BLOCK_SIZE   = 21     # neighborhood size (odd integer > 1)
C_VALUE      = 5      # constant subtracted from mean

# Morphological cleanup
KERNEL_SIZE = (3, 3)  # try (5,5) or (7,7) for stronger clean-up
MORPH_ITERS = 1

# Display settings
TILE_SIZE   = (400, 400)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 2.5
THICKNESS   = 6
TEXT_COLOR  = (0, 255, 0)   # bright green

# ----------------------------------------------------------------------------

def process_image(title, path):
    img_color = cv2.imread(path)
    if img_color is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    clahe    = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    img_clahe = clahe.apply(img_gray)

    img_filtered = cv2.bilateralFilter(
        img_clahe,
        BILAT_DIAM,
        BILAT_SIGMAC,
        BILAT_SIGMAS
    )

    mask = cv2.adaptiveThreshold(
        img_filtered,
        maxValue=255,
        adaptiveMethod=ADAPT_METHOD,
        thresholdType=THRESH_TYPE,
        blockSize=BLOCK_SIZE,
        C=C_VALUE
    )

    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
    mask_open  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=MORPH_ITERS)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERS)

    img_gray_bgr     = cv2.cvtColor(img_gray,     cv2.COLOR_GRAY2BGR)
    img_clahe_bgr    = cv2.cvtColor(img_clahe,    cv2.COLOR_GRAY2BGR)
    img_filtered_bgr = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
    mask_clean_bgr   = cv2.cvtColor(mask_clean,   cv2.COLOR_GRAY2BGR)


    cv2.putText(img_color,        f"{title} - Color",      (10, 100), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS)
    cv2.putText(img_gray_bgr,     f"{title} - Grayscale",  (10, 100), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS)
    cv2.putText(img_clahe_bgr,    f"{title} - CLAHE",      (10, 100), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS)
    cv2.putText(img_filtered_bgr, f"{title} - Bilateral",   (10, 100), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS)
    cv2.putText(mask_clean_bgr,   f"{title} - Clean Mask", (10, 100), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS)


    tiles = [
        cv2.resize(img_color,        TILE_SIZE),
        cv2.resize(img_gray_bgr,     TILE_SIZE),
        cv2.resize(img_clahe_bgr,    TILE_SIZE),
        cv2.resize(img_filtered_bgr, TILE_SIZE),
        cv2.resize(mask_clean_bgr,   TILE_SIZE)
    ]
    return tiles

rows = []
for scene, path in image_paths.items():
    rows.append(process_image(scene, path))

grid = [np.hstack(r) for r in rows]
full_display = np.vstack(grid)

cv2.imshow("Improved Adaptive Thresholding Results", full_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
