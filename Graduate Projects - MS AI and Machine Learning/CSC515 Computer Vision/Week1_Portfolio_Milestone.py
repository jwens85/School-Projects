import cv2 #(OpenCV Documentation, n.d.)
import os

#Show OpenCV version to verify installation
print("OpenCV version:", cv2.__version__)

#Specify path to Brain Image using a raw string (r"")
path = r"C:\Users\jwens\Desktop\CSUGlobal\CSC515\Brain_Image.jpg"

#Use imread(path) to load the image from the specified path
image = cv2.imread(path)

#Use imshow() with arguments for the window name and image specified above
cv2.imshow("Brain Image", image)
#So that the window doesn't immediately close cv2.waitKey(0) waits til any key is pressed
cv2.waitKey(0)

#os.path.expanduser("~") ensures that this script will work across different platforms
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "brain_copy.jpg")
#(Python Documentation, n.d.)

#Use imwrite() with arguments for desktop path and image that were defined above
cv2.imwrite(desktop_path, image)

#Print an output to signify a successful write using an f string
print(f"Image saved at: {desktop_path}")

#References
#GeeksforGeeks. (n.d.). OpenCV Python Tutorial.
#Retrieved March 23, 2025, from https://www.geeksforgeeks.org/opencv-python-tutorial/

#OpenCV Documentation. (n.d.). OpenCV Tutorials.
#Retrieved March 23, 2025, from https://docs.opencv.org/4.x/d9/df8/tutorial_root.html

#Python Documentation. (n.d.). os.path — Common pathname manipulations. Python 3.10.12 documentation.
#Retrieved March 23, 2025, from https://docs.python.org/3/library/os.path.html