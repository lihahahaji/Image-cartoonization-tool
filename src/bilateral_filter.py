import numpy as np

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
