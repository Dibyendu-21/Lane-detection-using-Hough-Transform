# Lane detection using Hough Transform
This repo gives a demonstration of detecting lanes using Hough transformation. Hough transform can easily detect straight lines from images after converting it from image space (cartesian coordinate) to hough space (polar coordinates). It has not only detected lanes in an image but also able to separate out the left and right lanes by coloring them  differently.

## Design Pipeline
The pipeline is demonstrated as follows:
* Convert original image to HSL
![HSL Image](/Output/HSL/Lane_1_hsl_image.png?raw=true)
* Isolate yellow and white from HSL image
### White isolated Image
![White isolated Image](/Output/Isolate%20White%20Pixel/Lane_1_hsl_isolate_white.png?raw=true)
### Yellow isolated Image
![Yellow isolated Image](/Output/Isolate%20Yellow%20Pixel/Lane_1_hsl_isolate_yellow.png?raw=true)
* Combine isolated HSL with original image
![Yellow and White isolated Image](/Output/Isolate%20White%20and%20Yellow/Lane_1_hsl_combined_isolated_image.png?raw=true)
* Convert image to grayscale for easier manipulation
![Grayscale Image](https://drive.google.com/uc?export=view&id=1pO6A64XHcAbaOvhSNphCQ2vOqIqspHsc)
* Apply Gaussian Blur to smoothen edges
![Blurred Image](/Output/Denoised/Lane_1_blur_isolated_image.png?raw=true)
* Apply Canny Edge Detection on smoothed gray image
![Edge Detection](Output/Edge%20Detection%20using%20Canny%20Filter/Lane_1_canny_isolated_image.png?raw=true)
* Trace Region Of Interest and discard all other lines identified by our previous step that are outside this region
![ROI](Output/ROI/Lane_1_canny_ROI_image.png?raw=true)
* Perform a Hough Transform to find lanes within our region of interest and trace them in red
![Lane Detection](Output/ROI/Lane_1_canny_ROI_image.png?raw=true)
* Separate left and right lane
![Lane Separation](Output/Separated%20Lanes/Lane_1_Different_Lane_Colors.png?raw=true)
