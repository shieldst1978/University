import cv2
import numpy as np


def unsharp_mask(image, sigma=1.2, amount=1.0):
    """Sharpen by subtracting a Gaussian blur from the image."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


# Load the image
img = cv2.imread("St_Pauls.jpg")
if img is None:
    raise FileNotFoundError("Could not load St_Pauls.jpg")

# Resize to 800px wide while maintaining aspect ratio
height, width = img.shape[:2]
new_width = 800
scale = new_width / width
new_height = int(height * scale)
resized = cv2.resize(img, (new_width, new_height))

# Method 1: Mild kernel (softer than the original [-1,-1,-1] variant)
# Increase the centre value (e.g. 6, 7, 8) for a stronger effect.
mild_kernel = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]],
    dtype=np.float32,
)
sharpened_kernel = cv2.filter2D(resized, -1, mild_kernel)

# Method 2: Unsharp mask — usually the most controllable approach.
# sigma  : how wide the blur is (higher = affects broader edges)
# amount : sharpening strength (try 0.5 to 2.0)
sharpened_unsharp = unsharp_mask(resized, sigma=1.2, amount=1.1)

# Display all versions for comparison
cv2.imshow("Original resized", resized)
cv2.imshow("Sharpened - mild kernel", sharpened_kernel)
cv2.imshow("Sharpened - unsharp mask", sharpened_unsharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save outputs
cv2.imwrite("St_Pauls_sharpened_kernel.jpg", sharpened_kernel)
cv2.imwrite("St_Pauls_sharpened_unsharp.jpg", sharpened_unsharp)
print("Saved St_Pauls_sharpened_kernel.jpg and St_Pauls_sharpened_unsharp.jpg")
