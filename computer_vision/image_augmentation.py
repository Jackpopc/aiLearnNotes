import cv2
import numpy as np
from skimage.util.dtype import convert


class ImageAugmented(object):
    def __init__(self, path="../data/2007_000129.jpg"):
        self.img = cv2.imread(path)
        self.h, self.w = self.img.shape[0], self.img.shape[1]

    # 1. 镜像变换
    def flip(self, flag="h"):
        generate_img = np.zeros(self.img.shape)
        if flag == "h":
            for i in range(self.h):
                for j in range(self.w):
                    generate_img[i, self.h - 1 - j] = self.img[i, j]
        else:
            for i in range(self.h):
                for j in range(self.w):
                    generate_img[self.h - 1 - i, j] = self.img[i, j]
        return generate_img

    # 2. 缩放
    def _resize_img(self, shape=(100, 300)):
        return cv2.resize(self.img, shape)

    # 3. 旋转
    def rotated(self):
        center = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), 45, 1)
        return cv2.warpAffine(self.img, center, (self.w, self.h))

    # 4. 平移
    def translation(self, x_scale=100, y_scale=100):
        move = np.float32([[1, 0, x_scale], [0, 1, y_scale]])
        return cv2.warpAffine(self.img, move, (self.w, self.h))

    # 5. 改变亮度
    def change_light(self, alpha=1.5, scale=3):
        return cv2.addWeighted(self.img, alpha, np.zeros(self.img.shape).astype(np.uint8), 1 - alpha, scale)

    # 6. 添加噪声
    def add_noise(self, mean=0, var=0.01):
        img = np.multiply(self.img, 1. / 255, dtype=np.float64)
        noise = np.random.normal(mean, var ** 0.5,
                                 img.shape)
        img = convert(img, np.floating)
        out = img + noise
        return out


if __name__ == '__main__':
    aug = ImageAugmented()
    img = aug.translation()
    cv2.imshow("img", img)
    cv2.waitKey()