---
title: 【动手学计算机视觉】第六讲：传统目标检测之Harris角点检测
---

## **前言**

在传统目标识别中，特征提取是最终目标识别效果好坏的一个重要决定因素，因此，在这项工作里，有很多研究者把主要精力都放在特征提取方向。在传统目标识别中，主要使用的特征主要有如下几类：

- 边缘特征
- 纹理特征
- 区域特征
- 角点特征

本文要讲述的Harris角点检测就是焦点特征的一种。

目前角点检测算法主要可归纳为3类：

- 基于灰度图像的角点检测
- 基于二值图像的角点检测
- 基于轮廓的角点检测

因为角点在现实生活场景中非常常见，因此，角点检测算法也是一种非常受欢迎的检测算法，尤其本文要讲的Harris角点检测，可以说传统检测算法中的经典之作。

## **Harris角点检测**

***什么是角点？***

要想弄明白角点检测，首先要明确一个问题，什么是角点？

![KfJDit.png](https://s2.ax1x.com/2019/10/29/KfJDit.png)

这个在现实中非常常见，例如图中标记出的飞机的角点，除此之外例如桌角、房角等。这样很容易理解，但是该怎么用书面的语言阐述角点？

角点就是轮廓之间的交点。

如果从数字图像处理的角度来描述就是：**像素点附近区域像素无论是在梯度方向、还是在梯度幅值上都发生较大的变化。**

这句话是焦点检测的关键，也是精髓，角点检测算法的思想就是由此而延伸出来的。

角点检测的算法思想是：选取一个固定的窗口在图像上以任意方向的滑动，如果灰度都有较大的变化，那么久认为这个窗口内部存在角点。

要想实现角点检测，需要用数学语言对其进行描述，下面就着重用数学语言描述一下角点检测算法的流程和原理。

用 $w(x,y)$ 表示窗口函数，  $[u,v]$为窗口平移量，像素在窗口内的变化量为，

 $$E(u, v)=\sum_{x, y} w(x, y)[I(x+u, y+v)-I(x, y)]^{2} $$

其中 $I(x, y) $为平移前的像素灰度值， $I(x+u, y+v)$ 为平移后的像素灰度值，

通过对灰度变化部分进行泰勒展开得到，

 $$\begin{array}{c}{\sum[I(x+u, y+v)-I(x, y)]^{2}} \\ {\approx \sum\left[I(x, y)+u I_{x}+v I_{y}-I(x, y)\right]^{2} \\ =\sum u^{2} I_{x}^{2}+2 u v I_{x} I_{y}+v^{2} I_{y}^{2} \\ =\sum \left[ \begin{array}{ll}{u} & {v}\end{array}\right] \left[ \begin{array}{l}{I_{x}^{2}} & {I_{x} I_{y}} \\ {I_{x} I_{y}} & {I_{y}^{2}}\end{array}\right] \left[ \begin{array}{l}{u} \\ {v}\end{array}\right] \\ = \left[ \begin{array}{ll}{u} & {v}\end{array}\right] (\sum\left[ \begin{array}{l}{I_{x}^{2}} & {I_{x} I_{y}} \\ {I_{x} I_{y}} & {I_{y}^{2}}\end{array}\right]) \left[ \begin{array}{l}{u} \\ {v}\end{array}\right]} \end{array}  $$

因此得到，

 $$E(u, v) \cong[u, v] M \left[ \begin{array}{l}{u} \\ {v}\end{array}\right] $$

矩阵 $M$ 中 $I_x$ 、 $I_y$ 分别是像素在 $x$ 、 $y$ 方向的梯度，从上述化简公式可以看出，灰度变化的大小主要取决于矩阵，

 $$M=\sum_{x, y} W(x, y) \left[ \begin{array}{cc}{I_{x}(x, y)^{2}} & {I_{x}(x, y) I_{y}(x, y)} \\ {I_{x}(x, y) I_{y}(x, y)} & {I_{y}(x, y)^{2}}\end{array}\right] $$

现在在回过头来看一下角点与其他类型区域的不同之处：

- 平坦区域：梯度方向各异，但是梯度幅值变化不大
- 线性边缘：梯度幅值改变较大，梯度方向改变不大
- 角点：梯度方向和梯度幅值变化都较大

明白上述3点之后看一下怎么利用其矩阵 $M$ 进行角点检测。

根据主成分分析(PCA)的原理可知，如果对矩阵 $M$ 对角化，那么，特征值就是主分量上的方差，矩阵是二维的方阵，有两个主分量，如果在窗口区域内是角点，那么梯度变化会较大，像素点的梯度分布比较离散，这体现在特征值上就是特征值比较大。

换句话说，

- 如果矩阵对应的两个特征值都较大，那么窗口内含有角点
- 如果特征值一个大一个小，那么窗口内含有线性边缘
- 如果两个特征值都很小，那么窗口内为平坦区域

读到这里就应该明白了，角点的检测转化为数学模型，就是求解窗口内矩阵的特征值并且判断特征值的大小。

如果要评价角点的强度，可以用下方公式，

$$R=\operatorname{det} M-k(\operatorname{trace} M)^{2}  \tag{1} $$

其中，

 $$\operatorname{det} M=\lambda_{1} \lambda_{2} $$

 $$\operatorname{trace} M=\lambda_{1}+\lambda_{2} $$

## **编程实践**

因为Harris角点检测算法非常经典，因此，一些成熟的图像处理或视觉库都会直接提供Harris角点检测的算法，以OpenCV为例，

```python
import cv2
import numpy as np

filename = '2007_000480.jpg'

img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,blockSize,ksize,k)

"""
其中，
gray为灰度图像，
blockSize为邻域窗口的大小，
ksize是用于Soble算子的参数，
k是一个常量，取值为0.04~0.06
"""
```

因为本文要讲一步一步实现Harris角点检测算法，因此，对OpenCV提供的函数不多阐述，下面开始一步一步实现Harris角点检测算法。

检点检测算法的流程如下：

1. 利用公式(1)求出输入图像每个位置的角点强度响应
2. 给定阈值，当一个位置的强度大于阈值则认为是角点
3. 画出角点

首先是第一步，根据上述提到的公式求矩阵的特征值和矩阵的迹，然后计算图像的角点强度，这里选取常数k=0.04，

```python
def calculate_corner_strength(img, scale=3, k=0.06):
    # 计算图像在x、y方向的梯度
    # 用滤波器采用差分求梯度的方式
    gradient_imx, gradient_imy = zeros(img.shape), zeros(img.shape)
    filters.gaussian_filter(img, (scale, scale), (0, 1), gradient_imx)
    filters.gaussian_filter(img, (scale, scale), (1, 0), gradient_imy)

    # 计算矩阵M的每个分量
    I_xx = filters.gaussian_filter(gradient_imx*gradient_imx, scale)
    I_xy = filters.gaussian_filter(gradient_imx*gradient_imy, scale)
    I_yy = filters.gaussian_filter(gradient_imy*gradient_imy, scale)

    # 计算矩阵的迹、特征值和响应强度
    det_M = I_xx * I_yy - I_xy ** 2
    trace_M = I_xx + I_yy
    return det_M + k * trace_M ** 2
```

接下来完成第2步，根据给定阈值，获取角点，

```python
def corner_detect(img, min=15, threshold=0.04):
    # 首先对图像进行阈值处理
    _threshold = img.max() * threshold
    threshold_img = (img > _threshold) * 1
    coords = array(threshold_img.nonzero()).T
    candidate_values = [img[c[0], c[1]] for c in coords]
    index = argsort(candidate_values)

    # 选取领域空间，如果邻域空间距离小于min的则只选取一个角点
    # 防止角点过于密集
    neighbor = zeros(img.shape)
    neighbor[min:-min, min:-min] = 1
    filtered_coords = []
    for i in index:
        if neighbor[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            neighbor[(coords[i, 0] - min):(coords[i, 0] + min),
            (coords[i, 1] - min):(coords[i, 1] + min)] = 0
    return filtered_coords
```

然后是画出角点，

```python
def corner_plot(image, filtered_coords):
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], 'ro')
    axis('off')
    show()
```

检测结果，

![KfJcQS.png](https://s2.ax1x.com/2019/10/29/KfJcQS.png)

完整代码如下，

```python
from scipy.ndimage import filters
import cv2
from matplotlib.pylab import *


class Harris(object):
    def __init__(self, img_path):
        self.img = cv2.imread(img_path, 0)

    def calculate_corner_strength(self):
        # 计算图像在x、y方向的梯度
        # 用滤波器采用差分求梯度的方式
        scale = self.scale
        k = self.k
        img = self.img
        gradient_imx, gradient_imy = zeros(img.shape), zeros(img.shape)
        filters.gaussian_filter(img, (scale, scale), (0, 1), gradient_imx)
        filters.gaussian_filter(img, (scale, scale), (1, 0), gradient_imy)

        # 计算矩阵M的每个分量
        I_xx = filters.gaussian_filter(gradient_imx*gradient_imx, scale)
        I_xy = filters.gaussian_filter(gradient_imx*gradient_imy, scale)
        I_yy = filters.gaussian_filter(gradient_imy*gradient_imy, scale)

        # 计算矩阵的迹、特征值和响应强度
        det_M = I_xx * I_yy - I_xy ** 2
        trace_M = I_xx + I_yy
        return det_M + k * trace_M ** 2

    def corner_detect(self, img):
        # 首先对图像进行阈值处理
        _threshold = img.max() * self.threshold
        threshold_img = (img > _threshold) * 1
        coords = array(threshold_img.nonzero()).T
        candidate_values = [img[c[0], c[1]] for c in coords]
        index = argsort(candidate_values)

        # 选取领域空间，如果邻域空间距离小于min的则只选取一个角点
        # 防止角点过于密集
        neighbor = zeros(img.shape)
        neighbor[self.min:-self.min, self.min:-self.min] = 1
        filtered_coords = []
        for i in index:
            if neighbor[coords[i, 0], coords[i, 1]] == 1:
                filtered_coords.append(coords[i])
                neighbor[(coords[i, 0] - self.min):(coords[i, 0] + self.min),
                (coords[i, 1] - self.min):(coords[i, 1] + self.min)] = 0
        return filtered_coords

    def corner_plot(self, img, corner_img):
        figure()
        gray()
        imshow(img)
        plot([p[1] for p in corner_img], [p[0] for p in corner_img], 'ro')
        axis('off')
        show()

    def __call__(self, k=0.04, scale=3, min=15, threshold=0.03):
        self.k = k
        self.scale = scale
        self.min = min
        self.threshold = threshold
        strength_img = self.calculate_corner_strength()
        corner_img = self.corner_detect(strength_img)
        self.corner_plot(self.img, corner_img)


if __name__ == '__main__':
    harris = Harris("2007_002619.jpg")
    harris()
```

---

> 我整理了一些计算机视觉、Python、强化学习、优化算法等方面的电子书籍、学习资料，同时还打包了一些我认为比较实用的工具，如果需要请关注公众号【平凡而诗意】，回复相应的关键字即可获取~

## 更多我的作品

[Jackpop：学习资源：图像处理从入门到精通](https://zhuanlan.zhihu.com/p/67343443)

[Jackpop：Python调试神器之PySnooper](https://zhuanlan.zhihu.com/p/67457275)

[Jackpop：【动手学计算机视觉】第五讲：传统目标检测之特征工程](https://zhuanlan.zhihu.com/p/66166633)

[Jackpop：是时候给大家推荐这款强大的神器了](https://zhuanlan.zhihu.com/p/66053762)


