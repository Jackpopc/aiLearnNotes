import cv2
from matplotlib import pyplot as plt


# 1.读取图像并转化为灰度图
img = cv2.imread("../data/2007_000793.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 2.显示灰度直方图
def histogram(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])
    plt.plot(range(len(hist)), hist)
    plt.show()
histogram(gray)

# 3.直方图均衡化
dst = cv2.equalizeHist(gray)
histogram(dst)

cv2.imshow("histogram", dst)
cv2.waitKey()