import numpy as np
from PIL import Image

# Make sure that the image you want to get output from is present
# in the same folder as the python file. Run this code using different images too :)
image_name = "WIN_20230326_10_12_27_Pro.png"

# Open the image file
img = Image.open(image_name)

# Convert the image to a numpy array and extract the grayscale values  (ravel() extracts the grayscale values )
img_array = np.array(img)
gray_values = img_array.ravel()
# Now numpy array gray_values contains gray value of each pixel.

##### TO DO #####

# Using Introduction.py see If image contains all pixel values are zero or not.
# Convert the given black image (Image is black because all grayscale values are near to zero) into
# img_stretched so that our eyes can see what's actually there in the image.
# This process is a real life application in Biology using Digital Image Processing.
# Change something in img_array :)
# Run your code with given 3 images by changing Image name in the code.

# HINTS
# 1. Increase the contrast (Read last week theory- Histogram Stretching)
# 2. min_value = np.min(gray_values) - This gives minimum value from the numpy array gray_values. Similarly you can find maximum value :)

# ACTIVITY
# I have given three images a.png, b.png, c.png. The time order of taking this image is as follows:
# a.png b.png c.png
#   t    t+T   t+2T
# Can you guess process?

#yes,the process is cell division.

#### WRITE YOUR CODE HERE ####
#histogram stretching
min_value = np.min(gray_values)
max_value = np.max(gray_values)
stretched_values = ((gray_values - min_value) / (max_value - min_value)) * 255

# Reshape the stretched values back into the shape of the original image
stretched_array = np.reshape(stretched_values, img_array.shape)

############# END ############

# Convert the numpy array back to an image
img_stretched = Image.fromarray(np.uint8(stretched_array))
img_stretched.save(f"final-{image_name}")
