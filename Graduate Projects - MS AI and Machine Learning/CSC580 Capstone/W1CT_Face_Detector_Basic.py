from PIL import Image, ImageDraw
import face_recognition
import os

# Absolute path to the input image
img_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/solvay.jpg"

# Confirm the file exists before loading
assert os.path.exists(img_path), f"File not found: {img_path}"

# Load image as a NumPy array
img_array = face_recognition.load_image_file(img_path)

# Detect face locations in the image
detected_faces = face_recognition.face_locations(img_array)
print(f"Detected {len(detected_faces)} face(s) in the image.")

# Prepare the image for drawing
visual = Image.fromarray(img_array)
overlay = ImageDraw.Draw(visual)

# Annotate each detected face with a red rectangle and print coordinates
for i, (top, right, bottom, left) in enumerate(detected_faces):
    print(f"Face {i+1}: Top={top}, Left={left}, Bottom={bottom}, Right={right}")
    overlay.rectangle([left, top, right, bottom], outline="red", width=2)

# Display the final annotated image
visual.show()
