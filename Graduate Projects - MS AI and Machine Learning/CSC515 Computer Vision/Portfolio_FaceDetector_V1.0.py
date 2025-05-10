import cv2

# List of full image file paths
image_file_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg"
]

# Load OpenCV's Haar cascade classifier for frontal face detection
frontal_face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Loop through each image path
for image_index, image_path in enumerate(image_file_paths):
    original_image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    detected_faces = frontal_face_classifier.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=5)

    # Draw bounding box around each detected face
    for (x_coordinate, y_coordinate, box_width, box_height) in detected_faces:
        cv2.rectangle(original_image, (x_coordinate, y_coordinate), (x_coordinate + box_width, y_coordinate + box_height), (0, 0, 255), 2)

    # Display the image with face detection results
    display_window_title = f"Image {image_index + 1} - Face Detection"
    cv2.imshow(display_window_title, original_image)

# Wait for user input and close all image display windows
cv2.waitKey(0)
cv2.destroyAllWindows()
