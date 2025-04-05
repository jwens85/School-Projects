import cv2

#Load the image from the local file path
image_path = r"C:\Users\jwens\Desktop\CSUGlobal\CSC515\Image_Puppy.jpg"
image = cv2.imread(image_path)

#Extract color channels as blue, green, red
blue, green, red = cv2.split(image)
red_matrix = red
green_matrix = green
blue_matrix = blue

#Display each color channel as grayscale
cv2.imshow("Red Matrix", red_matrix)
cv2.imshow("Green Matrix", green_matrix)
cv2.imshow("Blue Matrix", blue_matrix)

#Merge channels back into the original truecolor image
truecolor_image = cv2.merge((blue_matrix, green_matrix, red_matrix))
cv2.imshow("Truecolor Image (Original RGB)", truecolor_image)

#Swap Red and Green Channels to GRB
GRB_image = cv2.merge((blue_matrix, red_matrix, green_matrix))
cv2.imshow("GRB Image", GRB_image)

#Wait for key press and clean up
cv2.waitKey(0)
cv2.destroyAllWindows()
