import cv2 #Import OpenCV for functions like reading, transforming, and displaying images
import numpy as np #NumPy will be handling our array operations like matrix multiplication
from skimage.morphology import skeletonize #SciPy Toolkit's image processing library for skeletonization

#Define a function to take a file path to our fingerprint image that we will be applying morphologies to
#All of our morphologies will be tabbed under this function
def process_latent_fingerprint(image_file_path):
    #Load in the image as a single-channel grayscale using OpenCV's imread() function
    grayimage = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    #Basic error handling for bad paths or unsupported file formats
    if grayimage is None:
        raise FileNotFoundError("Image not found or cannot be read.")

    #Histogram equalization to improve local contrast between ridges and valleys, helps with smudges and fading
    #We set clipLimit=3.0 is a balanced parameter to enhance fingerprints without creating excessive noise
    #Lower clipLimit values would create less noise but be less effective, values higher than 3.0 seem to over-amplify noise
    #We set tileGridSize=(8,8) to set the number of regions the image is divided into for local contrast enhancement
    #Larger tiles (i.e. 4x4) will give more global contrast adjustment and less detail enhancement
    #Smaller tiles (i.e. 16x16) will give more local contrast adjustment but risk noise amplification and uneven brightness
    #(GeeksforGeeks, 2023)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(grayimage)

    #Declare the variable kernel to create a 5x5 elliptical structuring element for our morphological operations later in the code
    #OpenCV's getStructuringElement is a function provided to create a custom-shaped kernel, in this case, an ellipse
    #Other options for kernel shapes could be a rectangle (cv2.MORPH_RECT), or a cross (cv2.MORPH_CROSS)
    #An ellipse is appropriate here due to the more rounded structures found in fingerprint ridges
    #We'll set the kernel size as 5x5 for this, a balanced approach to remove small noise and bridge gaps between broken ridges
    #A smaller kernel (i.e. 3x3) would be gentler, good for preserving fine details but weak at removing noise
    #A larger (i.e. 9x9) would be more aggressive, better at removing noise but might eliminate or distort fine ridges
    #(Shimat, n.d.)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    #Here's the section where we'll apply the required morphological transformation to our CLAHE-enhanced fingerprint image
    #OpenCV's cv2.dilate() function is used to expand the white foreground in our binary or grayscale image
    #Dilation is useful for tasks like bridging broken fingerprint ridges, connecting features, and filling small gaps
    #(Morgan, 2023-b) has a pretty good guide to image dilation in OpenCV
    dilated_image = cv2.dilate(clahe_image, kernel, iterations=1)
    #OpenCV's cv2.erosion() function is the inverse of dilation, it shrinks the white regions and removes small noise
    #Erosion is useful for eliminating small specks that might get falsely identified as ridge patterns
    #(Morgan, 2023-a) has a pretty good guide to image erosion in OpenCV
    eroded_image = cv2.erode(clahe_image, kernel, iterations=1)
    #OpenCV's cv2.morphologyEx() function provides some more advanced morphological transformations by combining basic operations
    #Opening is useful for removing small foreground noise while preserving the shapes and sizes of larger structures
    #(TutorialsPoint, n.d.) describes opening as an "erosion followed by a dialation."
    #Arguments are cv2.morphologyEX(Image to be opened, Morphological Flag, and the structuring element)
    opened_image = cv2.morphologyEx(clahe_image, cv2.MORPH_OPEN, kernel)
    #Another cv2.morphologyEX() function is closing, can be thought of as the inverse of opening
    #Closing will close small dark spots (black pixels) in the white foreground. Useful to bridge ridges and fill gaps caused by smudges
    #Arguments are the same as above, except that we use cv2.MORPH_CLOSE here
    #(Sachdev, 2024) describes closing as a dilation followed by an erosion
    closed_image = cv2.morphologyEx(clahe_image, cv2.MORPH_CLOSE, kernel)

    #Skeletonization reduces objects in a binary image to a 1-pixel wide representation of a feature
    #This works by iteratively peeling layers off of pixels from the boundary of the white foreground until only the skeleton remains
    #Useful for preserving topological features that allow for feature identification while reducing visual clutter
    #We will need to import SkImage's skeletonize() library for this, which expects a binary input with white (1) foreground and black (0) background
    #For this assignment, we'll use `_` to discard the threshold value returned by cv2.threshold, since it’s not needed
    #We're only interested in the second output, the binarized image, which will be stored in the variable binary
    #The cv2.threshold() function converts a grayscale image into a binary image, with pixels set to a max (white) or a min (black) value
    #Arguments here are cv2.threshold(Input Image, Threshold Value, Max value, and Type)
    #Here, the threshold value is set to 0 and using Otsu's method, OpenCV will automatically compute the optimal value
    #Here, maxval needs to be set to 1 to pass the output to skeletonize(), all pixels above the Otsu threshold become 1 and all pixels below become 0
    #The thresholding technique Otsu is selected to automatically calculate the best threshold when combined with THRESH_BINARY
    #(Helmy, 2025) explains Otsu's method for thresholding and image segmentation
    _, binary = cv2.threshold(clahe_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #To match the style of our other morphological transforms, we will need to invert the binary using binary = 1 - binary
    #This flips the black and white pixels as all 1's become 0's and all 0's become 1's
    binary = 1 - binary
    #We will define skeleton as the output of the skeletonization algorithm on the binary image
    #The result of skeletonize() is a boolean NumPy array (true/false)
    #To convert to true = 1 and false = 0, we use astype(np.unit8) as OpenCV doesn't use bool images
    #NumPy's unit8 stands for an unsigned 8-bit integer, the datatype used to represent pixel values in OpenCV and NumPy
    #Now we use * 255 to make ridges white (255) and background black (0)
    #I needed help from the LLM on this one (Grimoire, 2025)
    skeleton = skeletonize(binary).astype(np.uint8) * 255

    #I thought it would look better if we could overlay the skeletonized fingerprint in orange on top of the CLAHE image
    #Again, (Grimoire, 2025) helped me here to get an orange outline over the CLAHE image to represent skeletonization
    #To convert the CLAHE image to a 3-channel BGR color, we used cv2.cvtColor(Image, cv2.COLOR_GRAY2BGR)
    base_color = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    #Defined overlay to create a blank image with the same shape and type as base_color, but completely black
    overlay = np.zeros_like(base_color)
    #For all pixels where the skeleton is white, assign the color 0, 85, 255 to give orange skeleton lines
    overlay[skeleton == 255] = [0, 85, 255]
    #Now we can stitch the two constructs together using cv2.addedWeighted(base_color, 1.0 weight for base and overlay, 0 gamma offset (no brightness adjustment))
    skeleton_overlay = cv2.addWeighted(base_color, 1.0, overlay, 1.0, 0)

    #If we want the skeletonization outline to appear as orange in the output, we will need to re-colorize back to 3-channel BGR images
    #We will define to_color(image) to return cv2.cvtColor(Image, and cv2's COLOR_GRAY2BGR) function
    def to_color(image): return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #Convert the original grayscale fingerprint to BGR so it can be used in color displays and side-by-side visualizations
    original_color = to_color(grayimage)
    #Apply the BGR color conversion to all transformed grayscale images (CLAHE, dilated, eroded, opened, closed)
    #The skeletonization result has already been colorized separately with an orange overlay
    clahe_color = to_color(clahe_image)
    dilated_image_color = to_color(dilated_image)
    eroded_image_color = to_color(eroded_image)
    opened_image_color = to_color(opened_image)
    closed_image_color = to_color(closed_image)

    #Prep the images by resizing and labeling with a descriptive text tag
    def prep(image, label):
        #Use cv2.resize() to create 250x250 standard images
        #We set interpolation=cv2.INTER_AREA to shrink or enlarge by taking an average of the neighboring pixels to reduce artifacts
        image = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA)
        #We can add labels using cv2.putText() to draw on the image
        #Arguments here are (Image, Label, (Position), Font, Font Scale, (Color), Thickness, and Anti Ailising)
        cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 100), 2, cv2.LINE_AA)
        #Return the results of prep() after resizing and labeling
        return image

    #Make a side by side comparison showing the original fingerprint image on the left, and the transformed version on the right
    #To do this we'll define the function make_block, with arguments for the processed image and its label
    def make_block(mod_image, label):
        #Horizontally stack multiple NumPy arrays side by side, and return the result
        #The hstack keyword is a built-in function within NumPy
        return np.hstack((
            #Call our prep function to prepare the original fingerprint image
            prep(original_color.copy(), "Original Image"),
            #Call our prep function to prepare the modified image, with its modification label
            prep(mod_image.copy(), label)
        ))
    #Create variables block1, block2, etc. to represent the different images we've transformed using morphologies
    block1 = make_block(clahe_color, "CLAHE Enhanced")
    block2 = make_block(dilated_image_color, "Dilated Image")
    block3 = make_block(eroded_image_color, "Eroded Image")
    block4 = make_block(opened_image_color, "Opening Applied")
    block5 = make_block(closed_image_color, "Closing Applied")
    block6 = make_block(skeleton_overlay, "Skeleton Overlay")

    #Vertical and Horizontal whitespace bufferes between the image blocks, to allow for side-by-side comparison with the original
    #I used the LLM (Grimoire, 2025) to fine-tune the spacing to look more presentable
    h_spacer = 255 * np.ones((block1.shape[0], 10, 3), dtype=np.uint8)
    v_spacer = 255 * np.ones((10, block1.shape[1] * 3 + 20, 3), dtype=np.uint8)
    row_top = np.hstack((block1, h_spacer, block2, h_spacer, block3))
    row_bot = np.hstack((block4, h_spacer, block5, h_spacer, block6))
    final_image = np.vstack((row_top, v_spacer, row_bot))

    #Create a name for the window that displays our results
    cv2.namedWindow("Fingerprint Morphology Results", cv2.WINDOW_NORMAL)
    #Use cv2.imshow to display our final image
    cv2.imshow("Fingerprint Morphology Results", final_image)
    #Don't close the window until a key is pressed
    cv2.waitKey(0)
    #When a key is pressed, close the window
    cv2.destroyAllWindows()
    #Save the image
    cv2.imwrite("Image_Fingerprint_Morphology_Results.png", final_image)

#Run the code directly, not imported as a module in another script. Could be removed if this program were to be exported
if __name__ == "__main__":
    input_path = r"C:\Users\jwens\Desktop\CSUGlobal\GitHub\Projects\Graduate Projects - MS AI and Machine Learning\CSC515 Computer Vision\Image_Fingerprint.png"
    process_latent_fingerprint(input_path)

#References
#GeeksforGeeks. (2023, May 19). CLAHE Histogram Equalization – OpenCV. GeeksforGeeks.
#Retrieved April 20, 2025, from https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/

#Grimoire. (2025, April 20). Chat about fingerprint skeletonization, OpenCV, and morphological operations [AI conversation]. ChatGPT.
#Retrieved April 20, 2025, from https://chat.openai.com/

#Helmy, B. E.-D. (2023, May 6). Understanding Otsu’s Method for Image Segmentation. Baeldung.
#Retrieved April 20, 2025, from https://www.baeldung.com/cs/otsu-segmentation

#Morgan, J. (2023, March 30). Applying erosion to images with OpenCV: A complete guide. Jeremy Morgan.
#Retrieved April 20, 2025, from https://www.jeremymorgan.com/tutorials/opencv/erosion-opencv-python/

#Morgan, J. (2023, March 31). Dilation Demystified: A Complete Guide to Image Dilation with OpenCV. Jeremy Morgan.
#Retrieved April 20, 2025, from https://www.jeremymorgan.com/tutorials/opencv/dilate-opencv-python/

#Sachdev, A. (2024, March 17). Closing (Morphological Operation) — Image Processing. Medium.
#Retrieved April 20, 2025, from https://medium.com/@anshul16/closing-morphological-operation-image-processing-59a0ef6210e3​

#Shimat. (n.d.). Cv2.GetStructuringElement Method (MorphShapes, Size). OpenCvSharp Documented Class Library.
#Retrieved April 20, 2025, from https://shimat.github.io/opencvsharp_docs/html/a83a9f62-e3dc-c3e9-36bd-1758462d2198.htm

#TutorialsPoint. (2021, March 17). Performing an opening operation on an image using OpenCV.
#Retrieved April 20, 2025, from https://www.tutorialspoint.com/performing-an-opening-operation-on-an-image-using-opencv

