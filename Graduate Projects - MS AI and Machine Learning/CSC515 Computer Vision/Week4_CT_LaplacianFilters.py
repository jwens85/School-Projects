import cv2
import numpy as np
import matplotlib.pyplot as subplot

#Define the variable noisyimage as the grayscale conversion of Image_Noisy.jpg
#Gaussian blurring and Laplacian edge detection work on pixel intensity (brightness)
#Using a single-channel grayscale image instead of an RGB 3-channel image makes the most sense
noisyimage = cv2.imread('Image_Noisy.jpg', cv2.IMREAD_GRAYSCALE)

#Set sigma value for standard deviation using Gaussian blur
#This controls how much smoothing is applied
#A small sigma (i.e. '1') might not suppress enough noise
#Since our kernel size is limited, sigma values above ~2 will have significant diminishng effect
Gaussian_σ = 2

#Define the variable as a list of kernel sizes, = 3 leads to a 3 x 3 matrix, etc. as specified in requirements
#For Gaussian blur, larger kernels blur more neighboring pixels resulting in stronger smoothing and lower noise
#For Laplacian edge detection, larger kernels are more sensitive to gradual edges but might miss small/sharp features
#This needs to be a list because we will use a for loop later in the program to iterate over the different kernel sizes
filter_window_sizes = [3, 5, 7]

#Create a 3 x 3 grid of subplots as outlined in the requirements.
#This is a function of MatPlotLib's PyPlot (MatPlotLib, n.d.)
#(3, 3) for the desired number of columns and rows
#The keyword argument figsize=(8, 8) controls how big the subplot appears on the screen.
#Later in the code, subplot_grid[row, column] will be called as an array to access the specific subplot desired
fig, subplot_grid = subplot.subplots(3, 3, figsize=(8, 8))

#Here is the for loop that runs the filter logic across the different kernel sizes
#We defined the kernels as a list earlier, so now we can use their index in the list as the subplot row
#We will declare the variable kernel_dimension here for the 3x3, 5x5, and 7x7 kernels required
#Use enumerate to access both the kernel size and its index for subplot row placement
for row_index, kernel_dimension in enumerate(filter_window_sizes):
    #Loop body applies the Gaussian filter to noisyimage with that specific kernel
    #Arguments for cv2.GaussianBlur are (source, (kernel size), sigma on the horizontal axix already declared as Gaussian_σ)
    #We will declare Gaussian_result here to store the output image after applying the Gaussian blur
    Gaussian_result = cv2.GaussianBlur(noisyimage, (kernel_dimension, kernel_dimension), Gaussian_σ)

    #Continuing the for loop body to apply only the Laplacian filter to noisyimage for edge detection
    #Arguments for cv2.Laplacian are (source, destination bit depth will be a 64-bit float, kernel size as kernel_dimension)
    #We will declare Laplacian_result here to store the output image after applying the Laplacian edge detection
    laplacian_raw = cv2.Laplacian(noisyimage, cv2.CV_64F, ksize=kernel_dimension)
    #We will need to convert the raw Laplacian output to an 8-bit unsigned image (OpenCV, n.d.)
    #This is necessary because the raw Laplacian output contains floats, which wouldn't be displayed properly using imshow()
    #We also need to account for the possibility that the raw Laplacian's output could include negative numbers
    #We can make all pixels positive by using cv2.convertScaleAbs to keep the edge strength as an absolute value
    laplacian_result = cv2.convertScaleAbs(laplacian_raw)

    #Apply Laplacian filter to the Gaussian-blurred image Gaussian_result
    #Same logic as before, just switching the input from noisyimage to Gaussian_result
    Laplacian_after_Gaussian = cv2.Laplacian(Gaussian_result, cv2.CV_64F, ksize=kernel_dimension)
    #Here we will declare combined_result to store the Laplacian of Gaussian (LoG), a well-known edge detection technique
    #There's a good article explaining this technique linked in the references below (Fisher et al., 2003)
    combined_result = cv2.convertScaleAbs(Laplacian_after_Gaussian)

    #Use OpenCV's imshow for the first column of Gaussian blurring results
    #We created a 3 x 3 subplot grid earlier in the code
    #The array subplot_grid[row, column] gives access to the specific subplot, here row_index is the current row number
    #This comes from looping through filter_window_sizes with enumerate(filter_window_sizes)
    #We will use imshow(Gaussian_result) to display the stored image in Gaussian_result in row 0, column 0
    #MatPlotLib defaults to a rainbow colormap, we could use black and white by adding imshow(Gaussian_result, camp='gray')
    #I think 'viridis' stands out best, but other colormap options are available (MatPlotLib, n.d.-a)
    subplot_grid[row_index, 0].imshow(Gaussian_result, cmap='viridis')
    #Set the title for the subplot with f-string arguments for (Title, dimension x dimension, and σ value)
    subplot_grid[row_index, 0].set_title(f'Gaussian {kernel_dimension}x{kernel_dimension}, σ={Gaussian_σ}')
    #This line controls axis labels for pixel positions, I think 'on' looks cleanest, but there are other options
    #Other options are 'off', 'equal', 'scaled', 'tight', 'image', or 'auto'
    subplot_grid[row_index, 0].axis('on')

    #Same logic as above, but for the Laplacian filtered image
    subplot_grid[row_index, 1].imshow(laplacian_result, cmap='viridis')
    subplot_grid[row_index, 1].set_title(f'Laplacian {kernel_dimension}x{kernel_dimension}')
    subplot_grid[row_index, 1].axis('on')

    #Same logic as above, but for the LoG image
    subplot_grid[row_index, 2].imshow(combined_result, cmap='viridis')
    subplot_grid[row_index, 2].set_title(f'Laplacian Over Gaussian {kernel_dimension}x{kernel_dimension}, σ={Gaussian_σ}')
    subplot_grid[row_index, 2].axis('on')

#Rename the window title for clarity, subplot.gcf() returns the current figure object we're working with
#Canvas is the drawing surface backend component that renders the figure
#Manager is the window manager that gives access to the GUI window displaying the figure
#We can set the window title using (obviously) .set_window_title
subplot.gcf().canvas.manager.set_window_title('Gaussian and Laplacian Filters for Different Kernel Windows')
#Enables automatic spacing adjustmentbetween subplots, other options are constrained_layout=True, or GridSpec for manual control
#Automatic spacing appears to be sufficient for this simple subplot layout
subplot.tight_layout()
#Render and display the figures and plots created in the script
subplot.show()

#References:
#Fisher, R., Perkins, S., Walker, A., & Wolfart, E. (2003). Laplacian/Laplacian of Gaussian. HIPR2: Hypermedia Image Processing Reference.
#Retrieved April 12, 2025, from https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm

#Matplotlib. (n.d.-a). Colormap reference. Matplotlib Documentation.
#Retrieved April 12, 2025, from https://matplotlib.org/stable/gallery/color/colormap_reference.html

#Matplotlib. (n.d.-b). matplotlib.pyplot.subplots. Matplotlib Documentation.
#Retrieved April 12, 2025, from https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

#OpenCV. (n.d.). cv::convertScaleAbs. OpenCV Documentation.
#Retrieved April 12, 2025, from https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga3460e9c9f37b563ab9dd550c4d8c4e7d