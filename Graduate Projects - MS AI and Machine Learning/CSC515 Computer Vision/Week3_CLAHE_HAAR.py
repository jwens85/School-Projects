import cv2

# Load the image
img = cv2.imread('Image_Faces.png')

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a CLAHE object with specific parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE to enhance local contrast
enhanced_img = clahe.apply(gray_img)

# Convert enhanced grayscale back to BGR for drawing colored circles/rectangles
enhanced_color = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

# Load pre-trained Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create a copy for detection results
detection_img = enhanced_color.copy()

# Detect faces
faces = face_cascade.detectMultiScale(enhanced_img, 1.3, 5, minSize=(30, 30))

# For each face detected
for (x, y, w, h) in faces:
    # Draw green circle around the face
    center = (x + w // 2, y + h // 2)
    radius = min(w, h) // 2
    cv2.circle(detection_img, center, radius, (0, 255, 0), 2)

    # Create a region of interest for the upper part of face only
    roi_gray = enhanced_img[y:y + h // 2, x:x + w]
    roi_color = detection_img[y:y + h // 2, x:x + w]

    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20), maxSize=(w // 3, h // 3))

    # Only keep top 2 eye detections if more are found
    if len(eyes) > 2:
        # Sort by size (area)
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

    # For each eye detected
    for (ex, ey, ew, eh) in eyes:
        # Draw red rectangle around the eye
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

# Display the results
cv2.imshow('Original Image', gray_img)
cv2.imshow('CLAHE Enhanced Image', enhanced_img)
cv2.imshow('Face and Eye Detection', detection_img)

# Save the detection result
cv2.imwrite('face_eye_detection.jpg', detection_img)

cv2.waitKey(0)
cv2.destroyAllWindows()