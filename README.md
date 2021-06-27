# Lane detection using Hough Transform
This repo gives a demonstration of detecting lanes using Hough transformation. Hough transform can easily detect straight lines from images after converting it from image space (cartesian coordinate) to hough space (polar coordinates). It has not only detected lanes in an image but also able to separate out the left and right lanes by coloring them  differently.

## The Pipeline
The pipeline is demonstrated as follows:
*	Convert original image to HSL
* Isolate yellow and white from HSL image
* Combine isolated HSL with original image
* Convert image to grayscale for easier manipulation
* Apply Gaussian Blur to smoothen edges
* Apply Canny Edge Detection on smoothed gray image
* Trace Region Of Interest and discard all other lines identified by our previous step that are outside this region
* Perform a Hough Transform to find lanes within our region of interest and trace them in red
* Separate left and right lanes
