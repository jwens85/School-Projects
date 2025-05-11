#Import necessary libraries
import cv2 #OpenCV for computer vision and image processing
import os #Allows access to image paths
import numpy as np #Allows for hstack() and vstack() to build the final image grid
import torch #Allows PyTorch over GPU to run the MTCNN model's pretrained weights
from facenet_pytorch import MTCNN #Multi-task Cascaded CNN face detector library
import warnings #Allows suppression of FutureWarnings from PyTorch's facenet module
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch") #Ignore FutureWarnings

#Define the image paths as a list so that they can be iterated in a for loop later
image_paths = [
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_A1.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_B.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_C.jpg",
    r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Portfolio_D1.jpg"
]

#Define path to Haar cascade directory
face_cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

#Function to calculate intersection over union (IoU) between 2 bounding boxes
#This will be useful later in the code to eliminate overlapping eye bounding boxes
def compute_intersection_over_union(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area
#(Rosebrock, 2016) (Grimoire, 2025)

#Uses Gaussian Blur to anonymize only areas where eyes are detected
def blur_eyes_detected(eyes_detected, kernel_size=(31, 31)): #Substantial blurring effect from large kernel size
    if eyes_detected.size:
        return cv2.GaussianBlur(eyes_detected, kernel_size, 0) #Only blur if region of interest contains pixel data
    return eyes_detected

#Process the MTCNN using GPU if available, if not, use CPU
computation_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Initialize MTCNN face detector from facenet_pytorch library with tuned parameters (Gradilla, 2020)
mtcnn_face_detector = MTCNN(
    keep_all=True, #Detect all faces in the image, not just the largest
    device=computation_device, #Use GPU if available, if not use CPU (See above)
    thresholds=[0.3, 0.3, 0.4], #Confidence thresholds for P-Net (proposal network), R-Net (refinement network) and O-Net (output network)
    min_face_size=15, #Ignore faces smaller than 15x15 pixels to reduce false positives
    factor=0.7 #Control image pyramid scaling ratio (Nath, 2024)
)

#Histogram Equalization using Open CV's CLAHE
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #Limits contrast amplification and uses an 8x8 grid for fine control

#Post-processing image control
processed_image_list = [] #Empty list to store processed images
display_target_size = (400, 300) #Output dimensions for processed images regardless of original image sizes
eye_blur_radius = 10 #A value of 10 means a 20 x 20 pixel box will be blurred around each landmark center

#Beginning of the main image processing loop
for image_file_path in image_paths: #Iterate over each image path in image_paths[...]
    input_image = cv2.imread(image_file_path) #Load the image from disk into a NumPy array in BGR format
    haar_eye_regions = [] #Initialize an empty list to store coordinates of eye regions detected with HAAR
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) #Convert from BGR to grayscale
    enhanced_image = CLAHE.apply(grayscale_image) #Apply CLAHE to the converted image
    enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0) #Apply Gaussian blur with a 5x5 kernel size and let OpenCV automatically calculate the appropriate standard deviation (sigmaX)
    detected_face_rectangles = face_cascade_classifier.detectMultiScale(enhanced_image, 1.1, 5) #HAAR cascade to detect faces with scale factor 1.1 and 5 min neighbours
    #(GeeksforGeeks, 2025) Goes into detail about detectMultiScale's parameters

    #Nested loop, to draw a red rectangle around the HAAR detected face
    for (face_x, face_y, face_width, face_height) in detected_face_rectangles: #Rectangle coordinates (x, y, width, height) used to define the face region of interest (ROI) for later eye detection
        #Draw a rectangle on the image using OpenCV
        cv2.rectangle(
            input_image, #Rectangle will be drawn on the original input image (modified in place)
            (face_x, face_y), #Coordinates of the top-left corner
            (face_x + face_width, face_y + face_height), #Calculate the bottom right corner using width and height
            (0, 0, 255), 2) #Color is BGR pure red, box is 2 pixels thick

        #HAAR cascade eye detector limited to the region of interest defined by the detected face
        haar_eye_detections = eye_cascade_classifier.detectMultiScale(
            enhanced_image[face_y:face_y + face_height, face_x:face_x + face_width], #Use the preprocessed image to extract the face ROI
            scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)) #Scale down 10% at each pyramid level, requires 3 overlapping detections to validate a candidate, minimum size 20x20 pixels to reduce false positives

        #Convert from face-local coordinates to global image coordinates (Grimoire, 2025)
        eye_candidate_boxes = [
            (
                face_x + eye_x,
                face_y + eye_y,
                face_x + eye_x + eye_width,
                face_y + eye_y + eye_height
            )
            #Loop through each detected eye box and convert to image-global coordinates, store the result in a new list
            for (eye_x, eye_y, eye_width, eye_height) in haar_eye_detections
        ]

        #Sort the list of eye boxes in ascending order, will be used later to filter overlapping boxes for more precise detections
        eye_candidate_boxes.sort(
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]) #Use a lambda function to sort eye boxes by area
        ) #(Grimoire, 2025)

        #Initialize an empty list to hold the set of eye boxes without significant overlap
        non_overlapping_eye_boxes = []

        #Iterate over the sorted list of eye boxes, sorted by area with the smallest first
        for eye_box in eye_candidate_boxes:
            #Uses any() to check whether the current eye box overlaps significantly (>0.3), if not, the box is accepted
            if not any(
                compute_intersection_over_union(eye_box, existing_box) > 0.3
                for existing_box in non_overlapping_eye_boxes
            ):
                non_overlapping_eye_boxes.append(eye_box) #Add the current eye box to the list of eye boxes if it passes the overlap test
            #(Grimoire, 2025)

            #Stop condition to limit number of accepted eye boxes to 2 per face (Grimoire, 2025)
            if len(non_overlapping_eye_boxes) == 2:
                break

        #Draw green rectangles over each of the non overlapping eye regions
        for (region_left, region_top, region_right, region_bottom) in non_overlapping_eye_boxes:
            cv2.rectangle(input_image, (region_left, region_top), (region_right, region_bottom), (0, 255, 0), 2)

            #Store the coordinates of the accepted region to avoid re-blurring the same eye if detected again by MTCNN
            haar_eye_regions.append((region_left, region_top, region_right, region_bottom))
            #Extract the eye ROI from the original image
            eyes_detected = input_image[region_top:region_bottom, region_left:region_right]
            #Replace the eye area in the original image with the blurred version
            input_image[region_top:region_bottom, region_left:region_right] = blur_eyes_detected(eyes_detected)

    #Convert the image from BGR to RGB for PyTorch requirements
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    #Runn MTCNN on the image and return face bounding boxes, confidence score for each detection, and coordinates for facial landmarks
    detection_boxes, detection_probabilities, detection_landmarks = mtcnn_face_detector.detect(
        rgb_image,
        landmarks=True #If landmarks=True, the function returns the 5 key landmarks per face (eyes, nose, corners of mouth)
    )
    #(Timsler, 2020)

    #Initialize an empty list to store accepted face detections by MTCNN
    accepted_detection_indices = []

    #Filter MTCNN face detections based on confidence and draw a blue rectangle around accepted faces
    if detection_boxes is not None and detection_probabilities is not None:
        #Loop through bounding boxes and their confidence scores
        for detection_index, (detection_box, detection_probability) in enumerate(zip(detection_boxes, detection_probabilities)):
            #Only accept faces with at least 85% confidence
            if detection_probability < 0.85:
                continue
            #Draw a blue box around faces using our converted integer coordinates
            x_start, y_start, x_end, y_end = map(int, detection_box)
            cv2.rectangle(input_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2) #BGR Blue with 2 pixel thickness
            #Save the index of a valid detection so that its landmarks can be referenced later
            accepted_detection_indices.append(detection_index)

    #Blur any additional eyes detected by MTCNN that were not covered by Haar cascade
    if detection_landmarks is not None: #Ensures that landmarks were returned by MTCNN and avoids errors if no landmarks exist
        for detection_index in accepted_detection_indices: #Iterate over high confidence detections
            for landmark_point in detection_landmarks[detection_index][:2]: #Loop through the first two landmarks of each face (0 = left eye and 1 = right eye)
                eye_x, eye_y = map(int, landmark_point) #Convert eye coordinates from a floaat to an int for image indexing
                #Skip the landmark if it's already blurred by HAAR
                if any(
                    (region_left <= eye_x <= region_right and region_top <= eye_y <= region_bottom)
                    for (region_left, region_top, region_right, region_bottom) in haar_eye_regions
                ):
                    continue #(Grimoire 2025)
                #Define a square region of interest centered at the eye, size is based on eye_blur_radius, already defined as 10 (20 x 20)
                region_top_crop = max(0, eye_y - eye_blur_radius)
                region_bottom_crop = min(input_image.shape[0], eye_y + eye_blur_radius)
                region_left_crop = max(0, eye_x - eye_blur_radius)
                region_right_crop = min(input_image.shape[1], eye_x + eye_blur_radius)
                #Extract and blur the region around the MTCNN eye center
                eyes_detected = input_image[region_top_crop:region_bottom_crop, region_left_crop:region_right_crop]
                input_image[region_top_crop:region_bottom_crop, region_left_crop:region_right_crop] = blur_eyes_detected(eyes_detected)

    #Draw blue dots at eye landmarks detected by MTCNN if confidence is >85%
    if detection_landmarks is not None: #Ensures landmarks were returned and prevents errors if features were not found
        for detection_index in accepted_detection_indices: #Loop over the high confidence face detections
            for landmark_point in detection_landmarks[detection_index][:2]:#Access the first two landmarks for the eyes
                eye_x, eye_y = map(int, landmark_point) #Convert float landmark coordinates to ints for pixel indexing
                cv2.circle(input_image, (eye_x, eye_y), 3, (255, 0, 0), -1) #Draw a 3 pixel circle (dot) at each eye center

    #Resize the processed image into a fixed display size and append the resized image to the processed_image_list
    processed_image_list.append(cv2.resize(input_image, display_target_size))

#Display results in a 2×2 grid
top_row_image = np.hstack(processed_image_list[:2]) #Horizontally stack images 0 and 1 side by side on the top row
bottom_row_image = np.hstack(processed_image_list[2:]) #Horizontally stack images 2 and 3 side by side on the bottom row
image_grid = np.vstack([top_row_image, bottom_row_image]) #Stack the top and bottom row on top of each other
cv2.imshow("Faces with Blurred Eyes", image_grid) #Open a window called Faces with Blurred Eyes on our 2x2 grid
cv2.waitKey(0) #Keep the windows open until a key is pressed
cv2.destroyAllWindows() #If a key is pressed, close the window

#References:
#GeeksforGeeks. (2025, April 17). Face Detection using Cascade Classifier using OpenCV – Python.
#https://www.geeksforgeeks.org/face-detection-using-cascade-classifier-using-opencv-python/

#Gradilla, R. (2020, July 27). Multi-task Cascaded Convolutional Networks (MTCNN) for face detection and facial landmark alignment. Medium.
#https://medium.com/@iselagradilla94/multi-task-cascaded-convolutional-networks-mtcnn-for-face-detection-and-facial-landmark-alignment-7c21e8007923

#Grimoire. (2025). Face and Eye Detector [LLM conversation]. OpenAI ChatGPT.
#https://chat.openai.com/

#Nath, S. (2024, August 10). How MTCNN detects faces: A simple guide to powerful technology. LinkedIn.
#https://www.linkedin.com/pulse/how-mtcnn-detects-faces-simple-guide-powerful-technology-sujal-nath-lea7c

#Rosebrock, A. (2016, November 7). Intersection over Union (IoU) for object detection. PyImageSearch.
#https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

#Timesler. (2020). Guide to MTCNN in facenet-pytorch. Kaggle.
#https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch

