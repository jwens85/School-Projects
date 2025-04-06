import cv2

#File paths for HAAR XML Models and Selfie Photo
path_face_model = r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\haarcascade_frontalface_default.xml"
path_eye_model = r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\haarcascade_eye.xml"
path_input_photo = r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Selfie.jpg"

#Load detection models (OpenCV, 2013)
face_detector = cv2.CascadeClassifier(path_face_model)
eye_detector = cv2.CascadeClassifier(path_eye_model)

#Load and resize the image
original_img = cv2.imread(path_input_photo)

#Error handling for incorrect paths
if original_img is None:
    print("Could not load image. Check the path.")
    exit()

#Resize original image for 600 pixel width while keeping the original aspect ratio
desired_width = 600 #Desired width is 600 pixels
resize_ratio = desired_width / original_img.shape[1] #Calculate the scaling factor based on width
resized_img = cv2.resize(original_img, (desired_width, int(original_img.shape[0] * resize_ratio)))#Resize image height
#(Moukthika, 2025)

#Convert to grayscale for detection
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

#Scan the image and detect any faces that can be found using face_detector HAAR
#Argument for scaleFactor shrinks the image by 10% on each scale, balancing speed and accuracy
#Argument for minNeighbors sets minimum for overlapping rectangles to be considered a face balancing sensitivity with false positives
found_faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
#(Bispo, 2024)

#Iterate over all detected faces and calculate the center point about which the circle will be drawn
for (fx, fy, fw, fh) in found_faces:
    center_point = (fx + fw // 2, fy + fh // 2)
    #(siromer, 2024)

    #Drawing the circle around the detected face (elipse)
    cv2.ellipse(resized_img, center_point, (fw // 2, fh // 2), 0, 0, 360, (0, 255, 0), 2)
    #(axis lengths), rotation angle, start angle, end angle, (BRR Color code for green), thickness
    #(Geeksforgeeks, n.d.)

    #Define regions of interest (roi) on the grayscale image for eye detection, and the color image for eye rectangles
    roi_gray_face = gray_img[fy:fy+fh, fx:fx+fw] #fy for vertical range of face, fx for horizontal range of face
    roi_color_face = resized_img[fy:fy+fh, fx:fx+fw] #And the same again for the colorized image

    #Detect eyes inside the face region using HAAR Cascade eye_detector only looking inside the face region (roi_gray_face)
    detected_eyes = eye_detector.detectMultiScale(roi_gray_face, scaleFactor=1.1, minNeighbors=5)#same arguments as face detector
    #(PythonProgramming, n.d.)

    #Run a for loop to go through every detected eye and draw a red box around it
    for (ex, ey, ew, eh) in detected_eyes:#For every eye detected show me its position and size
        cv2.rectangle(roi_color_face, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        #Draw a rectangle(area to draw on face, (start at the top left corner), (draw to the bottom right corner), (BGR format), (Thickness))

#Draw text over the top of the image using Hershey fonts in putText (EDUCBA, 2023)
label_text = "This is Me" #Text to add
font_type = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX #Hershey font selection
font_size = 1.0 #Font size
text_color = (0, 0, 0) #Text color (Black)
font_thickness = 2 #Line thickness (thicker)

#Calculate size (width and height) of the text string using our already defined sizes and font
label_size, _ = cv2.getTextSize(label_text, font_type, font_size, font_thickness) #Use _ to ignore the baseline argument as it's not needed
label_x = (resized_img.shape[1] - label_size[0]) // 2 #Calculate the horizontal position to start drawing text (centered)
label_y = 50 #Calculate the vertical position to start drawing text (high)

#Line to add putText to resized_img
cv2.putText(resized_img, label_text, (label_x, label_y), font_type, font_size, text_color, font_thickness)
#(EDUCBA, 2023)

# Display the final result
cv2.imshow("Week 3 Critical Thinking Option #1", resized_img) #Display resized_img
cv2.waitKey(0) #Wait for keypress before closing window (no timer)
cv2.destroyAllWindows() #Close all windows after keypress

#References
#Bispo, N. (2024, February 1). Face detection made easy with OpenCV and Python. Django Unleashed. Retrieved April 6, 2025, from:
#https://medium.com/django-unleashed/face-detection-made-easy-with-opencv-and-python-8386c2e2701d

#EDUCBA. (2023, April 6). OpenCV putText. EDUCBA. Retrieved April 6, 2025, from:
#https://www.educba.com/opencv-puttext/

#GeeksforGeeks. (n.d.). Python OpenCV | cv2.ellipse() method. GeeksforGeeks. Retrieved April 6, 2025, from:
#https://www.geeksforgeeks.org/python-opencv-cv2-ellipse-method/

#Moukthika. (2025, March 10). Resizing and rescaling images with OpenCV. OpenCV. Retrieved April 6, 2025, from:
#https://opencv.org/blog/resizing-and-rescaling-images-with-opencv/

#OpenCV. (OpenCV, 2013). HAAR cascades directory. GitHub. Retrieved April 6, 2025, from:
#https://github.com/opencv/opencv/tree/master/data/haarcascades

#PythonProgramming (n.d.). Haar cascade object detection face & eye tutorial. PythonProgramming.net. Retrieved April 6, 2025, from:
#https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
