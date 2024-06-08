import cv2
import numpy as np
from matplotlib import pyplot as plt
def apply_bilateral_filter(image, diameter, sigma_i, sigma_s):
    return cv2.bilateralFilter(image, diameter, sigma_s, sigma_i)

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )
    return edges

def combine_edges_and_smooth(image, edges):
    # Convert edges to color
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Combine edges and smoothed image
    return cv2.bitwise_and(image, edges)

def cartoonize_image(image, diameter=9, sigma_i=75, sigma_s=75):
    # Apply bilateral filter
    smoothed_image = apply_bilateral_filter(image, diameter, sigma_i, sigma_s)
    
    # Detect edges
    edges = detect_edges(image)
    
    # Combine edges and smoothed image
    cartoon_image = combine_edges_and_smooth(smoothed_image, edges)
    
    return cartoon_image

# Read the input image
input_image = cv2.imread('12313.png')
if input_image is None:
    raise ValueError("Could not open or find the image.")

# Apply cartoon effect
cartoon_image = cartoonize_image(input_image)

# Save the result
cv2.imwrite('cartoon_image.jpg', cartoon_image)

# # Display the result
# cv2.imshow('Cartoon Image', cartoon_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 显示原始图像
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image)
plt.axis('off')  # 不显示坐标轴

# 显示去噪后的图像
plt.subplot(1, 2, 2)
plt.title('cartoonize Image')
plt.imshow(cartoon_image)
plt.axis('off')

# 显示图像
plt.show()