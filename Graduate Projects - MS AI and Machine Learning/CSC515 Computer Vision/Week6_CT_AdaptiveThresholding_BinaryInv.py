#Import statements for the libraries I will be using
import cv2 #Import OpenCV
import numpy as np #Import NumPy

#A dictionary is used here for key:value pairs for the image paths, this will allow me to iterate over the images later
image_paths = {
    "Indoor": r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Inside.jpg",
    "Outdoor": r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Outside.jpg",
    "Closeup": r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Closeup.jpg"
}

#A list is used here for processed images, which will start out empty until some images get processed
processed_images = []

#Define the process_image function with title and path as parameters
def process_image(title, path):
    #Load the input image from the path in BGR format using OpenCV's imread function
    image_color = cv2.imread(path)
    #Convert the color image from BGR to grayscale
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    #Apply Gaussian blur to the grayscale image for noise reduction before thresholding
    image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    #Apply adaptive mean thresholding to segment the blurred image
    #Binary threshold (inverted) sets a pixel to black if intensity is > threshold
    #Binary seems to work better than TRUNC or TOZERO based on experimentation
    #Block size 35 and subtraction constant 8 seems to be the fine-tuning that works best for these images
    image_thresh = cv2.adaptiveThreshold(image_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,35,8)

    #Labels for the images, using a dictionary here keeps the code cleaner than having to repeat for every transformation
    images_to_label = {
        f'{title} - Color': image_color,
        f'{title} - Grayscale': cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR),
        f'{title} - Blurred': cv2.cvtColor(image_blurred, cv2.COLOR_GRAY2BGR),
        f'{title} - Segmented': cv2.cvtColor(image_thresh, cv2.COLOR_GRAY2BGR)
    }

    #Label appearance settings
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 10
    text_color = (0, 165, 255) #Orange

    #Create an empty list for resized images
    resized_images = []

    #Iterate over each labeled image, apply the text label, resize to 400x400 pixels
    for label, img in images_to_label.items():
        cv2.putText(img, label, (10, 100), label_font, font_scale, text_color, thickness)
        resized = cv2.resize(img, (400, 400))
        resized_images.append(resized)

    #Store the processed and labeled images
    processed_images.append(resized_images)

#Iterate through each scene name and image path to process the images
for scene, path in image_paths.items():
    process_image(scene, path)

#Use a horizontal stack for the processed images to show them side by side
row0 = np.hstack(processed_images[0])
row1 = np.hstack(processed_images[1])
row2 = np.hstack(processed_images[2])

#Use a vertical stack to aggregate the 3 rows we just made
full_display = np.vstack((row0, row1, row2))

#Use OpenCV's imshow function to open the results in a separate window
cv2.imshow("Adaptive Thresholding Results", full_display)
cv2.waitKey(0)
cv2.destroyAllWindows()