import cv2
import numpy as np

# Load the original image
image = cv2.imread('Image_Money.jpg')

# Step 1: Scaling Transformation (2x Bigger)
scale_x = 2.0
scale_y = 2.0
scaled_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)

# Step 2: Rotation Transformation (95 Degrees Clockwise)
(h, w) = scaled_image.shape[:2]
center = (w // 2, h // 2)
angle = -95  # Negative for clockwise rotation

# Generate the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# Calculate new dimensions and apply rotation
cos = abs(rotation_matrix[0, 0])
sin = abs(rotation_matrix[0, 1])
new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))
rotation_matrix[0, 2] += (new_w / 2) - center[0]
rotation_matrix[1, 2] += (new_h / 2) - center[1]
rotated_image = cv2.warpAffine(scaled_image, rotation_matrix, (new_w, new_h))

# Step 3: Translation Transformation to Center the Gold Ink Jar
# Use the coordinates you clicked on: (68, 79)
gold_jar_x, gold_jar_y = 68, 79  # Clicked coordinates

# Desired center point (image center)
desired_center_x, desired_center_y = new_w // 2, new_h // 2

# Calculate translation distances
tx = desired_center_x - gold_jar_x
ty = desired_center_y - gold_jar_y

# Define the translation matrix to move the gold ink jar to the center
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation
translated_image = cv2.warpAffine(rotated_image, translation_matrix, (new_w, new_h))

# Step 4: Click to Capture Four Points for Perspective Transformation
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(translated_image, (x, y), 5, (0, 0, 255), -1)  # Draw red dot
        cv2.imshow("Select Four Corners", translated_image)
        if len(points) == 4:
            cv2.destroyAllWindows()

# Display the translated image and set mouse callback
cv2.imshow("Select Four Corners", translated_image)
cv2.setMouseCallback("Select Four Corners", click_event)
cv2.waitKey(0)

# Step 5: Apply Perspective Transformation if Four Points are Collected
if len(points) == 4:
    src_points = np.float32(points)

    # Define destination points to create a rectangular output
    width = 600
    height = 400
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Calculate the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    perspective_image = cv2.warpPerspective(translated_image, perspective_matrix, (width, height))

    # Show the final perspective-corrected image
    cv2.imshow("Perspective Corrected Image", perspective_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Four points not captured.")
