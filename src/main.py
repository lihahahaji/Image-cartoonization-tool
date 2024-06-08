import cv2
import numpy as np
from matplotlib import pyplot as plt
from cartoonize import cartoonize_image

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

# 显示原始图像
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image)
plt.axis('off')  # 不显示坐标轴

# 显示去噪后的图像
plt.subplot(1, 2, 2)
plt.title('Cartoonized Image')
plt.imshow(output_image)
plt.axis('off')

# 显示图像
plt.show()
