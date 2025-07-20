from PIL import Image, ImageDraw
import face_recognition
import os

#Absolute path to the photo
photo_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/solvay.jpg"

#This line checks whether the file specified by photo_path returns True if the file exists
#and returns False otherwise. If the file does not exist the statement stops execution and raises
#an AssertionError with a customized message indicating that the input image was not found.
#This assures that the script fails early and explicitly if the required image is missing
#If the file exists, the script continues as intended.
assert os.path.exists(photo_path), f"Input image not found: {photo_path}"

#This line assigns the variable frame to the image data loaded from the file path. This function
#call reads the image file located at photo_path and converts it into a NumPy array, which
#is the standard format for image data in Python. By loading the image as a NumPy array, the script
#can efficiently process the image's pixel data for tasks such as face detection, manipulation,
#or analysis.
frame = face_recognition.load_image_file(photo_path)

#These lines use the face_recognition library to detect all faces present in the image. The function
#analyzes the NumPy array stored in frame and returns a list of bounding boxes, where each box
#represents the coordinates of a detected face within the image. This list is assigned to the
#variable face_boxes. The print statement then outputs the total number of faces detected by displaying
#the length of the face_boxes list. This provides immediate feedback in the console output as to
#how many faces were found in the provided image, allowing the user to confirm that the face
#detection step is working as intended
face_boxes = face_recognition.face_locations(frame)
print(f"Total faces found: {len(face_boxes)}")

#Now we're ready to convert the image data from a NumPy array into a format that can be easily
#manipulated for drawing operations. The function takes the NumPy array frame and creates a PIL
#image object called canvas. This allows the script to use PIL's drawing utilities for annotating
#the image. The next line creates a drawing context associated with the canvas image. This drawing
#context stored in the drawer variable provides methods for drawing shapes, lines, or text
#directly onto the image, making it possible to add visual annotations such as bounding boxes
#around the detected faces.
canvas = Image.fromarray(frame)
drawer = ImageDraw.Draw(canvas)

#Simple for loop that iterates over each detected face in the image and draws a red bounding box
#around it. The enumerate function returns the index and the list contains the bounding box
#coordinates for each detected face. For every face, the script prints the index (starting at 1)
#and the specific coordinates top, left, bottom, and right as tuples to define the bounding box.
#The drawer.rectangle function is then called to draw a rectangle on the image using these
#coordinates, with a red outline and a line width of 2 pixels. This process visually highlights
#each detected face in the image, making it clear where the face recognition system has identified
#human faces.
for index, (top, right, bottom, left) in enumerate(face_boxes):
    print(f"Face {index + 1}: (Top={top}, Left={left}, Bottom={bottom}, Right={right})")
    drawer.rectangle([left, top, right, bottom], outline="red", width=2)

#Display the annotated image to the user. After all detected faces have been highlighted with
#red bounding boxes, this line opens the modified image in the system's default image viewer.
#This allows the user to visually inspect the results and confirm that faces have been correctly
#detected and marked in the image
canvas.show()
