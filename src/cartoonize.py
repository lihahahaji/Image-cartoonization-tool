import cv2
import numpy as np
from bilateral_filter.py import apply_bilateral_filter
from edge_detection.py import detect_edges

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
