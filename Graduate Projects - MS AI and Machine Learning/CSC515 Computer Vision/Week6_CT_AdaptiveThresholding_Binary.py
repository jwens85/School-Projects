import cv2
import numpy as np

image_paths = {
    "Indoor": r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Inside.jpg",
    "Outdoor": r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Outside.jpg",
    "Closeup": r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Closeup.jpg"
}

processed_images = []

def process_image(title, path):
    img_color = cv2.imread(path)
    if img_color is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    img_gray_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_blurred_color = cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2BGR)
    img_thresh_color = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 6
    text_color = (0, 255, 0)
    cv2.putText(img_color, f'{title} - Color', (10, 100), label_font, font_scale, text_color, thickness)
    cv2.putText(img_gray_color, f'{title} - Grayscale', (10, 100), label_font, font_scale, text_color, thickness)
    cv2.putText(img_blurred_color, f'{title} - Blurred', (10, 100), label_font, font_scale, text_color, thickness)
    cv2.putText(img_thresh_color, f'{title} - Thresholded', (10, 100), label_font, font_scale, text_color, thickness)
    resized_color = cv2.resize(img_color, (400, 400))
    resized_gray = cv2.resize(img_gray_color, (400, 400))
    resized_blurred = cv2.resize(img_blurred_color, (400, 400))
    resized_thresh = cv2.resize(img_thresh_color, (400, 400))
    processed_images.append([resized_color, resized_gray, resized_blurred, resized_thresh])

for scene, path in image_paths.items():
    process_image(scene, path)

row1 = np.hstack(processed_images[0])
row2 = np.hstack(processed_images[1])
row3 = np.hstack(processed_images[2])
full_display = np.vstack((row1, row2, row3))

cv2.imshow("Adaptive Thresholding Results", full_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
