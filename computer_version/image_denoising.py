import cv2
import numpy as np
import skimage
from skimage.util.dtype import convert

# 1.读取图像
img = cv2.imread("../data/2007_001458.jpg")

# 2.添加噪声
# 方法1：用第三方工具添加噪声
noise_img = skimage.util.random_noise(img, mode="salt")


# 方法2：用numpy生成噪声
# def add_noise(img):
#     img = np.multiply(img, 1. / 255,
#                         dtype=np.float64)
#     mean, var = 0, 0.01
#     noise = np.random.normal(mean, var ** 0.5,
#                              img.shape)
#     img = convert(img, np.floating)
#     out = img + noise
#     return out
# noise_img = add_noise(img)
# gray_img =  cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)

# 3.图像去噪
# 方法1：用第三方工具去噪
# denoise = cv2.medianBlur(img, ksize=3)
# denoise = cv2.fastNlMeansDenoising(img, ksize=3)
# denoise = cv2.GaussianBlur(img, ksize=3)

def compute_pixel_value(img, i, j, ksize, channel):
    h_begin = max(0, i - ksize // 2)
    h_end = min(img.shape[0], i + ksize // 2)
    w_begin = max(0, j - ksize // 2)
    w_end = min(img.shape[1], j + ksize // 2)
    return np.median(img[h_begin:h_end, w_begin:w_end, channel])

def denoise(img, ksize):
    output = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j, 0] = compute_pixel_value(img, i, j, ksize, 0)
            output[i, j, 1] = compute_pixel_value(img, i, j, ksize, 1)
            output[i, j, 2] = compute_pixel_value(img, i, j, ksize, 2)
    return output

# output = denoise(noise_img, 3)
cv2.imshow("noise_img", noise_img)
cv2.waitKey()