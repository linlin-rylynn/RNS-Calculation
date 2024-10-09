import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_snr(image):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯滤波器进行平滑，减少高频噪声
    smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 二值化处理：将黑色斑点变为白色区域（背景为黑色）
    _, binary_image = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找到所有轮廓，表示所有黑色斑点区域
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建与图像同样大小的掩码
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # 将所有黑色斑点区域填充为白色

    # 计算信号区域均值（所有黑色斑点的区域）
    signal_region = cv2.bitwise_and(gray_image, gray_image, mask=mask)  # 只保留黑色斑点区域
    signal_mean = np.mean(signal_region[mask == 255])  # 只计算掩码区域的均值

    # 背景区域：使用反掩码，去除所有黑色斑点
    background_mask = cv2.bitwise_not(mask)
    background_region = cv2.bitwise_and(gray_image, gray_image, mask=background_mask)

    # 过滤背景中的异常值（如极端高亮或低亮区域）
    background_values = background_region[background_mask == 255]
    background_values = background_values[background_values < np.percentile(background_values, 95)]  # 排除最高5%的值
    background_std = np.std(background_values)

    # 计算SNR
    snr = signal_mean / background_std

    # 保存信号区域和背景区域的图像
    cv2.imwrite('signal_region.png', signal_region)  # 保存黑色斑点区域
    cv2.imwrite('background_region.png', background_region)  # 保存背景区域图像

    return snr, signal_region, background_region

# 读取图像（彩色图像）
image = cv2.imread('500.tif')

# 计算SNR
snr_value, signal_region, background_region = calculate_snr(image)
print(f"The calculated SNR is: {snr_value:.2f}")

# 像素与实际尺寸转换，500像素对应于58毫米
pixel_to_mm_ratio = 58 / 500

# 获取图像尺寸
height, width = image.shape[:2]

# 使用matplotlib显示原图、信号区域和背景区域，且显示实际尺寸标注
plt.figure(figsize=(15, 10))

# 设置字体大小和格式
plt.rcParams.update({
    'font.size': 20,           # 修改字体大小
    'font.family': 'Arial',    # 修改字体格式为 Arial
    'axes.titlesize': 20,      # 标题字体大小
    'axes.labelsize': 20,      # 轴标签字体大小
    'xtick.labelsize': 20,     # x轴刻度字体大小
    'ytick.labelsize': 20,      # y轴刻度字体大小
    'axes.linewidth': 2.0  # 修改图表边线的粗细
})



# 显示原图
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original image')
plt.xlabel(f'x (mm)')
plt.ylabel(f'y (mm)')
plt.tick_params(width=2)

# 显示信号区域
plt.subplot(1, 3, 2)
plt.imshow(signal_region, cmap='gray')
plt.title('Signal region')
plt.xlabel(f'x (mm)')
plt.ylabel(f'y (mm)')
plt.tick_params(width=2)

# 显示背景区域
plt.subplot(1, 3, 3)
plt.imshow(background_region, cmap='gray')
plt.title('Background region')
plt.xlabel(f'x (mm)')
plt.ylabel(f'y (mm)')
plt.tick_params(width=2)

plt.tight_layout()
plt.show()
