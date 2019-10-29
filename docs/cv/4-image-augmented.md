## **前言**

近几年深度学习的大规模成功应用主要的就是得益于数据的累积和算例的提升，虽然近几年很多研究者竭力的攻克半监督和无监督学习，减少对大量数据的依赖，但是目前数据在计算机视觉、自然语言处理等人工智能技术领域依然占据着非常重要的地位。甚至可以说，大规模的数据是计算机视觉成功应用的前提条件。但是由于种种原因导致数据的采集变的十分困难，因此图像增广技术就在数据的准备过程中占据着举足轻重的角色，本文就概括一下常用的图像增广技术并编程实现相应手段。

## **介绍**

图像增广（image augmentation）技术通过对训练图像做一系列随机改变，来产生相似但又不同的训练样本，从而扩大训练数据集的规模。图像增广的另一种解释是，随机改变训练样本可以降低模型对某些属性的依赖，从而提高模型的泛化能力。

目前常用的图像增广技术有如下几种：

- 镜像变换
- 旋转
- 缩放
- 裁剪
- 平移
- 亮度修改
- 添加噪声
- 剪切
- 变换颜色

在图像增广过程中可以使用其中一种手段进行扩充，也可以使用其中的几种方法进行组合使用，由于概念比较简单，容易理解，所以接下来就边实现，边详细阐述理论知识。

## **几何变换**

首先以水平镜像为例，假设在原图中像素的坐标为，在镜像变换之后的图像中的坐标为，原图像坐标和镜像变换后的坐标之间的关系式：

 $$ \left\{ \begin{aligned} x_1 &=& w-1-x_0 \\ y_1 &=& y_0 \end{aligned} \right. $$

其中 ![w](https://www.zhihu.com/equation?tex=w)w 为图像的宽度。

那么两张图像的关系就是：

 $$\left[ \begin{array}{c}{x_{1}} \\ {y_{1}} \\ {1}\end{array}\right]=\left[ \begin{array}{ccc}{-1} & {0} & {w-1} \\ {0} & {1} & {0} \\ {0} & {0} & {1}\end{array}\right] \left[ \begin{array}{l}{x_{0}} \\ {y_{0}} \\ {1}\end{array}\right] $$

它的逆变换就是

 $$\left[ \begin{array}{c}{x_{0}} \\ {y_{0}} \\ {1}\end{array}\right]=\left[ \begin{array}{ccc}{-1} & {0} & {w-1} \\ {0} & {1} & {0} \\ {0} & {0} & {1}\end{array}\right] \left[ \begin{array}{l}{x_{1}} \\ {y_{1}} \\ {1}\end{array}\right] $$

从原图到水平镜像的变换矩阵就是：

 $$\left[ \begin{array}{ccc}{-1} & {0} & {w-1} \\ {0} & {1} & {0} \\ {0} & {0} & {1}\end{array}\right] $$

同理，可知，垂直镜像变换的关系式为：

 $$\left[ \begin{array}{ccc}{-1} & {0} & {w-1} \\ {0} & {1} & {0} \\ {0} & {0} & {1}\end{array}\right] $$

其中为图像高度。

通过上述可以知道，**平移**变换的数学矩阵为：

$$H=\left[ \begin{array}{lll}{1} & {0} & {d_{x}} \\ {0} & {1} & {d_{y}} \\ {0} & {0} & {1}\end{array}\right] $$

 其中和分别是像素在水平和垂直方向移动的距离。

![Kf3dZn.png](https://s2.ax1x.com/2019/10/29/Kf3dZn.png)

同理可以推广到旋转变换上，加上原像素的坐标为 ![(x_0,y_0)](https://www.zhihu.com/equation?tex=(x_0%2Cy_0))(x_0,y_0) ，该像素点相对于原点的角度为，假设有一个半径为的圆，那么原像素的坐标可以表示为：

 $$\left\{\begin{array}{l}{x_{0}=r \cos \alpha} \\ {y_{0}=r \cos \alpha}\end{array}\right. $$

加上旋转后的像素坐标为 ![(x_1,y_1)](https://www.zhihu.com/equation?tex=(x_1%2Cy_1))(x_1,y_1) ，旋转角度为 ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)\theta ，那么可以表示为：

 $$\left\{\begin{array}{l}{x_1=r \cos (\alpha+\theta)} \\ {y_1=r \sin (\alpha+\theta)}\end{array}\right. $$

通过展开、化简可得，

 $$\left\{\begin{array}{l}{x_1=r \cos (\alpha+\theta)=r \cos \alpha \cos \theta-r \sin \alpha \sin \theta=x_{0} \cos \theta-y_{0} \sin \theta} \\ {y_1=r \sin (\alpha+\theta)=r \sin \alpha \cos \theta+r \cos \alpha \sin \theta=x_{0} \sin \theta+y_{0} \cos \theta}\end{array}\right. $$

把上述公式写成数学矩阵形式为：

 $$\left[ \begin{array}{l}{x_1} \\ {y_1} \\ {1}\end{array}\right]=\left[ \begin{array}{ccc}{\cos \theta} & {-\sin \theta} & {0} \\ {\sin \theta} & {\cos \theta} & {0} \\ {0} & {0} & {1}\end{array}\right] \left[ \begin{array}{l}{x_{0}} \\ {y_{0}} \\ {1}\end{array}\right] $$

因此旋转变换的矩阵为：

 $$H=\left[ \begin{array}{ccc}{\cos \theta} & {-\sin \theta} & {0} \\ {\sin \theta} & {\cos \theta} & {0} \\ {0} & {0} & {1}\end{array}\right] $$

其他的几何变换方式和上述提到的核心思想大同小异，因此，就不再详细展开，感兴趣的可以在网上搜集一下，或者看一下数字图像处理相关的书籍，关注这些内容的讲解有很多。

## **编程实践**

> 编程实践过程中主要用到opencv、numpy和skimage。

读取图像：

```python
# 1. 读取图像
img = cv2.imread("./data/000023.jpg")
cv2.imshow("Origin", img)
cv2.waitKey()
```

![Kf3zJf.png](https://s2.ax1x.com/2019/10/29/Kf3zJf.png)

初始化一个矩阵，用于存储转化后的图像，

```python
generate_img = np.zeros(img.shape)
```

**1.水平镜像**

遍历图像的像素，用前文提到的数学关系式进行像素的转化，

```python
for i in range(h):
    for j in range(w):
        generate_img[i, w - 1 - j] = img[i, j]

cv2.imshow("Ver", generate_img.astype(np.uint8))
cv2.waitKey()
```

![Kf8kes.png](https://s2.ax1x.com/2019/10/29/Kf8kes.png)

> 备注：初始化的图像数据类型是numpy.float64，用opencv显示时无法正常显示，因此在显示时需要用astype(np.uint8)把图像转化成numpy.uint8数据格式。

**2.垂直镜像**

垂直镜像变换代码，

```python
for i in range(h):
    for j in range(w):
        generate_img[h-1-i, j] = img[i, j]
```

![Kf8eYV.png](https://s2.ax1x.com/2019/10/29/Kf8eYV.png)

> 镜像变换也可以直接调用opencv的flip进行使用。

**3.图像缩放**

这个比较简单，直接调用opencv的resize函数即可，

```python
output = cv2.resize(img, (100, 300))
```

![Kf8KlF.png](https://s2.ax1x.com/2019/10/29/Kf8KlF.png)

**4.旋转变换**

这个相对复杂一些，需要首先用getRotationMatrix2D函数获取一个旋转矩阵，然后调用opencv的warpAffine仿射函数安装旋转矩阵对图像进行旋转变换，

```python
center = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
rotated_img = cv2.warpAffine(img, center, (w, h))
```

![Kf8QOJ.png](https://s2.ax1x.com/2019/10/29/Kf8QOJ.png)

**5. 平移变换**

首先用numpy生成一个平移矩阵，然后用仿射变换函数对图像进行平移变换，

```python
move = np.float32([[1, 0, 100], [0, 1, 100]])
move_img = cv2.warpAffine(img, move, (w, h))
```

![Kf8USO.png](https://s2.ax1x.com/2019/10/29/Kf8USO.png)

**6.亮度变换**

亮度变换的方法有很多种，本文介绍一种叠加图像的方式，通过给原图像叠加一副同样大小，不同透明度的全零像素图像来修改图像的亮度，

```python
alpha = 1.5
light = cv2.addWeighted(img, alpha, np.zeros(img.shape).astype(np.uint8), 1-alpha, 3)
```

其中alpha是原图像的透明度，

![Kf8wOH.png](https://s2.ax1x.com/2019/10/29/Kf8wOH.png)

**7.添加噪声**

首先写一下噪声添加的函数，原理就是给图像添加一些符合正态分布的随机数，

```python
def add_noise(img):
    img = np.multiply(img, 1. / 255,
                        dtype=np.float64)
    mean, var = 0, 0.01
    noise = np.random.normal(mean, var ** 0.5,
                             img.shape)
    img = convert(img, np.floating)
    out = img + noise
    return out
```

![Kf8cff.png](https://s2.ax1x.com/2019/10/29/Kf8cff.png)

**8.组合变换**

除了以上方法单独使用之外，还可以叠加其中多种方法进行组合使用，比如可以结合选择、镜像进行使用，

![Kf850s.png](https://s2.ax1x.com/2019/10/29/Kf850s.png)

完整代码如下：

```python
import cv2
import numpy as np
from skimage.util.dtype import convert


class ImageAugmented(object):
    def __init__(self, path="./data/000023.jpg"):
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
                    generate_img[self.h-1-i, j] = self.img[i, j]
        return generate_img

    # 2. 缩放
    def _resize_img(self, shape=(100, 300)):
        return cv2.resize(self.img, shape)
    
    # 3. 旋转
    def rotated(self):
        center = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), 45,1)
        return cv2.warpAffine(self.img, center, (self.w, self.h))
    
    # 4. 平移
    def translation(self, x_scale=100, y_scale=100):
        move = np.float32([[1, 0, x_scale], [0, 1, y_scale]])
        return cv2.warpAffine(self.img, move, (self.w, self.h))
    
    # 5. 改变亮度
    def change_light(self, alpha=1.5, scale=3):
        return cv2.addWeighted(self.img, alpha, np.zeros(self.img.shape).astype(np.uint8), 1-alpha, scale)
    
    # 6. 添加噪声
    def add_noise(self, mean=0, var=0.01):
        img = np.multiply(self.img, 1. / 255, dtype=np.float64)
        noise = np.random.normal(mean, var ** 0.5,
                                 img.shape)
        img = convert(img, np.floating)
        out = img + noise
        return out
```

------

## 往期回顾

[Jackpop：【动手学计算机视觉】第一讲：图像预处理之图像去噪](https://zhuanlan.zhihu.com/p/57521026)

[Jackpop：【动手学计算机视觉】第二讲：图像预处理之图像增强](https://zhuanlan.zhihu.com/p/57537622)

[Jackpop：【动手学计算机视觉】第三讲：图像预处理之图像分割](https://zhuanlan.zhihu.com/p/60847136)

------

> 感兴趣的可以关注一下，也可以关注公众号"平凡而诗意"，我在公众号共享了一些资源和学习资料，关注后回复相应关键字可以获取。


