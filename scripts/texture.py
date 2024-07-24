import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt

# Load the color image
image_path = 'C:/Users/User/Desktop/image.png'  # Replace with your image path
color_image = cv2.imread(image_path)
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Convert the color image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

# Function to compute LBP
def compute_lbp(image, radius=1, n_points=8):
    lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp

# Compute LBP for texture
lbp_image = compute_lbp(gray_image)

# Plot the results
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
ax = axes.ravel()

ax[0].imshow(color_image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(gray_image, cmap='gray')
ax[1].set_title('Grayscale Image')
ax[1].axis('off')

ax[2].imshow(lbp_image, cmap='gray')
ax[2].set_title('Texture Image')
ax[2].axis('off')

plt.show()
