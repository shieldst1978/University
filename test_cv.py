import cv2

# Load the image
img = cv2.imread("St_Pauls.jpg")

# Resize the image to a width of 800 pixels while maintaining the aspect ratio
height, width = img.shape[:2]
new_width = 800
scale = new_width / width
new_height = int(height * scale)

# Convert the resized image to greyscale
resized = cv2.resize(img, (new_width, new_height))
grey_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Display the transformed image
cv2.imshow("Black and white and reduced size", grey_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()



