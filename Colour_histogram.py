import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



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

# Convert the resized image from Blue-Green-Red (BGR) to Red-Green-Blue (RGB) colour space
resized = cv2.resize(img, (new_width, new_height))
rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# Convert the original image to greyscale
grey_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Calculate histograms for each colour channel
colours = ("r", "g", "b")
for i, colour in enumerate(colours):
    histogram = cv2.calcHist([rgb_resized], [i], None, [256], [0, 256])
    plt.plot(histogram, color=colour)
    plt.xlim([0, 256])

# Create the histogram plot
plt.title("RGB Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Save the histogram plot to a file
output_path = script_dir / "histogram.png"
plt.tight_layout()
plt.savefig(str(output_path), dpi=300)
print(f"Histogram saved to: {output_path}")
plt.show()
    