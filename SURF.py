import cv2
from pathlib import Path
import numpy as np

# Load the image
script_dir = Path(__file__).resolve().parent
image_path = script_dir / "St_Pauls.jpg"
if not image_path.exists():
    image_path = script_dir / "images" / "St_Pauls.jpg"

# Check if the image was loaded successfully
img = cv2.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f"Could not load image at: {image_path}")

# Resize the image to a width of 800 pixels while maintaining the aspect ratio
height, width = img.shape[:2]
new_width = 800
scale = new_width / width
new_height = int(height * scale)
resized = cv2.resize(img, (new_width, new_height))

# Convert the original image to greyscale
grey_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(400)
keypoints, descriptors = surf.detectAndCompute(grey_resized, None)

# Draw the keypoints on the original image
keypoint_img = cv2.drawKeypoints(grey_resized, keypoints, img)

# Draw rich keypoints with size and orientation
rich_keypoint_img = cv2.drawKeypoints(grey_resized, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save the images with keypoints to files
output_path = script_dir / "surf_keypoints.jpg"
cv2.imwrite(str(output_path), keypoint_img)
print(f"SURF keypoints image saved to: {output_path}")
output_path_rich = script_dir / "surf_rich_keypoints.jpg"
cv2.imwrite(str(output_path_rich), rich_keypoint_img)
print(f"SURF rich keypoints image saved to: {output_path_rich}")

# create a numPy array of the keypoint coordinates
print("Keypoint coordinates:")
for kp in keypoints:
    print(kp.pt)
