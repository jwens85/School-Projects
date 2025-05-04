import cv2
import numpy as np
from sklearn.metrics import precision_score

def create_image(bg=255, shape=0):
    img = np.full((500, 500), bg, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), shape, -1)
    cv2.circle(img, (350, 350), 75, shape, -1)
    return img

def add_noise(img, std=10):
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * img.size * 0.5).astype(int)

    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]

    noisy[coords_salt[0], coords_salt[1]] = 255
    noisy[coords_pepper[0], coords_pepper[1]] = 0
    return noisy

def apply_morph_filter(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def get_edges(img, condition_label=None):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    canny = cv2.Canny(img_blur, 100, 200)

    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel_mag, 0, 255))
    _, sobel = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)

    if condition_label == "Noisy Low Contrast":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        lap_blur = cv2.GaussianBlur(enhanced, (5, 5), 1.2)
        lap = cv2.Laplacian(lap_blur, cv2.CV_64F, ksize=3)
        lap = np.uint8(np.clip(np.abs(lap), 0, 255))
        _, lap = cv2.threshold(lap, 60, 255, cv2.THRESH_BINARY)
    else:
        lap = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)
        lap = np.uint8(np.clip(np.abs(lap), 0, 255))
        _, lap = cv2.threshold(lap, 100, 255, cv2.THRESH_BINARY)

    return canny, sobel, lap

def label_image(img, label):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img

def resize_image(img, size=300):
    return cv2.resize(img, (size, size))

def prepare_labeled_resized(img, label):
    return resize_image(label_image(img, label))

def create_ground_truth():
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (200, 200), 255, 1)
    cv2.circle(mask, (350, 350), 75, 255, 1)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

def run_tests(selected_tests, selected_images):
    test_cases = [
        {"bg": 255, "shape": 0, "noise": False, "label": "Clean High Contrast"},
        {"bg": 255, "shape": 0, "noise": True,  "label": "Noisy High Contrast"},
        {"bg": 200, "shape": 50, "noise": True,  "label": "Noisy Medium Contrast"},
        {"bg": 150, "shape": 100, "noise": True, "label": "Noisy Low Contrast"},
        {"bg": 255, "shape": 0, "noise": "salt_pepper", "label": "Salt and Pepper Noise + Morphology"},
    ]

    grid_rows = []
    gt = create_ground_truth()

    for i in selected_tests:
        case = test_cases[i]
        img = create_image(case["bg"], case["shape"])
        if case["noise"] == "salt_pepper":
            img = add_salt_and_pepper_noise(img)
            img = apply_morph_filter(img)
        elif case["noise"]:
            img = add_noise(img)

        canny, sobel, lap = get_edges(img, condition_label=case["label"])

        print(f"\n[{case['label']}]")
        print(f" Canny     Precision: {precision_score((gt > 0).flatten(), (canny > 0).flatten()):.2f}")
        print(f" Sobel     Precision: {precision_score((gt > 0).flatten(), (sobel > 0).flatten()):.2f}")
        print(f" Laplacian Precision: {precision_score((gt > 0).flatten(), (lap > 0).flatten()):.2f}")

        row = []
        if selected_images in ("input", "all"):
            row.append(prepare_labeled_resized(img, f"{case['label']} - Input"))
        if selected_images in ("canny", "all"):
            row.append(prepare_labeled_resized(canny, "Canny"))
        if selected_images in ("sobel", "all"):
            row.append(prepare_labeled_resized(sobel, "Sobel"))
        if selected_images in ("laplacian", "all"):
            row.append(prepare_labeled_resized(lap, "Laplacian"))

        if row:
            grid_rows.append(cv2.hconcat(row))

    if grid_rows:
        full_grid = cv2.vconcat(grid_rows)
        cv2.imshow("Edge Detection Results", full_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    print("Which image types would you like to display?")
    print("Options: input, canny, sobel, laplacian, all")
    selected_images = input("Enter one: ").strip().lower()

    print("\nWhich test(s) would you like to run?")
    print("1 - Clean High Contrast")
    print("2 - Noisy High Contrast")
    print("3 - Noisy Medium Contrast")
    print("4 - Noisy Low Contrast")
    print("5 - All Tests")
    print("6 - Salt and Pepper Noise + Morphology")
    test_input = input("Enter 1–6: ").strip()

    if test_input == "5":
        selected_tests = [0, 1, 2, 3, 4]
    elif test_input == "6":
        selected_tests = [4]
    elif test_input in {"1", "2", "3", "4"}:
        selected_tests = [int(test_input) - 1]
    else:
        print("Invalid test selection.")
        return

    if selected_images not in {"input", "canny", "sobel", "laplacian", "all"}:
        print("Invalid image type selection.")
        return

    run_tests(selected_tests, selected_images)

if __name__ == "__main__":
    main()
