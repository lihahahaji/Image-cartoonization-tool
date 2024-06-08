import cv2
import numpy as np

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
