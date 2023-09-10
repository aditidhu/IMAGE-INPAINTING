# Thresholding in Image Processing

Thresholding is a common segmentation technique used for separating an object from its background in image processing. This technique works by comparing each pixel of an image with a predefined threshold value and then dividing the pixels into two groups: one group with values higher than the threshold value and the other with values lower than the threshold value.

## Code Example using OpenCV

Below is a Python code example using the OpenCV library to perform thresholding on an image:

```python
import cv2
import numpy as np

# Load an image
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Define a threshold value (you can adjust this value)
threshold_value = 128

# Apply binary thresholding
ret, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Show the original and thresholded images
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
