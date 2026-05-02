import cv2
import numpy as np

# Load the image
img = cv2.imread("St_Pauls.jpg")

# Resize the image to a width of 800 pixels while maintaining the aspect ratio
height, width = img.shape[:2]
new_width = 800
scale = new_width / width
new_height = int(height * scale)
resized = cv2.resize(img, (new_width, new_height))

# Apply sharpening filter to the image
sharpening_kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
sharpened_image = cv2.filter2D(resized, -1, sharpening_kernel)

# Convert the resized image to greyscale
grey_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Display the transformed images
cv2.imshow("Black and white and reduced size", grey_resized)
cv2.imshow("Sharpened image", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# output the transformed images to files
cv2.imwrite("St_Pauls_bw.jpg", grey_resized)
cv2.imwrite("St_Pauls_sharpened.jpg", sharpened_image)
