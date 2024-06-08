import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_bilateral_filter(image, diameter, sigma_i, sigma_s):
    """
    手动实现双边滤波器，平滑图像，同时保留边缘。
    :param image: 输入图像
    :param diameter: 滤波窗口直径
    :param sigma_i: 像素值域的标准差
    :param sigma_s: 空间域的标准差
    :return: 滤波后的图像
    """
    rows, cols = image.shape[:2]
    half_diameter = diameter // 2
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # 预计算空间权重
    gauss_spatial = np.zeros((diameter, diameter), dtype=np.float32)
    for i in range(diameter):
        for j in range(diameter):
            x = i - half_diameter
            y = j - half_diameter
            gauss_spatial[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))

    # 对每个像素进行双边滤波
    for ch in range(3):  # 对每个通道
        for i in range(rows):
            for j in range(cols):
                Wp = 0
                filtered_pixel = 0

                for k in range(-half_diameter, half_diameter + 1):
                    for l in range(-half_diameter, half_diameter + 1):
                        ni = i + k
                        nj = j + l

                        if 0 <= ni < rows and 0 <= nj < cols:
                            intensity_diff = image[ni, nj, ch] - image[i, j, ch]
                            gauss_range = np.exp(-(intensity_diff**2) / (2 * sigma_i**2))
                            weight = gauss_spatial[k + half_diameter, l + half_diameter] * gauss_range
                            Wp += weight
                            filtered_pixel += weight * image[ni, nj, ch]

                filtered_image[i, j, ch] = filtered_pixel / Wp

    return filtered_image

def detect_edges(image):
    """
    使用自适应阈值方法检测图像的边缘。
    :param image: 输入图像
    :return: 检测到的边缘图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_uint8 = (gray * 255).astype(np.uint8)
    gray_blurred = cv2.medianBlur(gray_uint8, 7)
    edges = cv2.adaptiveThreshold(
        gray_blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )
    return edges

def combine_edges_and_smooth(image, edges):
    """
    将平滑后的图像与边缘图像合并，生成卡通化效果。
    :param image: 平滑后的图像
    :param edges: 边缘图像
    :return: 卡通化图像
    """
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored = edges_colored / 255.0  # 归一化到 [0, 1]
    combined_image = np.clip(image * edges_colored, 0, 1)
    return combined_image

def cartoonize_image(image, diameter=9, sigma_i=75, sigma_s=75):
    """
    对图像应用卡通化效果。
    :param image: 输入图像
    :param diameter: 滤波窗口直径
    :param sigma_i: 像素值域的标准差
    :param sigma_s: 空间域的标准差
    :return: 卡通化图像
    """
    # 应用双边滤波
    smoothed_image = apply_bilateral_filter(image, diameter, sigma_i, sigma_s)
    
    # 检测边缘
    edges = detect_edges(image)
    
    # 合并平滑图像和边缘图像
    cartoon_image = combine_edges_and_smooth(smoothed_image, edges)
    
    return cartoon_image

# 读取输入图像
input_image = cv2.imread('test.png')
if input_image is None:
    raise ValueError("无法打开或找到图像。")

input_image = input_image.astype(np.float32) / 255.0  # 归一化到 [0, 1]

# 应用卡通化效果
cartoon_image = cartoonize_image(input_image)

# 保存结果
output_image = (cartoon_image * 255).astype(np.uint8)
cv2.imwrite('cartoon_image.jpg', output_image)

# # 显示结果
# cv2.imshow('Cartoon Image', output_image)
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
plt.imshow(output_image)
plt.axis('off')

# 显示图像
plt.show()