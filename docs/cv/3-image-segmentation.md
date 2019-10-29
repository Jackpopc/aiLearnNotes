---
title: 【动手学计算机视觉】第三讲：图像预处理之图像分割
---

## 前言

图像分割是一种把图像分成若干个独立子区域的技术和过程。在图像的研究和应用中，很多时候我们关注的仅是图像中的目标或前景(其他部分称为背景)，它们对应图像中特定的、具有独特性质的区域。为了分割目标，需要将这些区域分离提取出来，在此基础上才有可能进一步利用，如进行特征提取、目标识别。因此，图像分割是由图像处理进到图像分析的关键步骤，在图像领域占据着至关重要的地位。

## 介绍

提到图像分割，主要包含两个方面：

- 非语义分割
- 语义分割

首先，介绍一下非语义分割。

非语义分割在图像分割中所占比重更高，目前算法也非常多，研究时间较长，而且算法也比较成熟，此类图像分割目前的算法主要有以下几种：

- 阈值分割

![KfleJO.png](https://s2.ax1x.com/2019/10/29/KfleJO.png)

阈值分割是图像分割中应用最多的一类，该算法思想比较简单，给定输入图像一个特定阈值，如果这个阈值可以是灰度值，也可以是梯度值，如果大于这个阈值，则设定为前景像素值，如果小于这个阈值则设定为背景像素值。

阈值设置为100对图像进行分割：

![Kfl1eI.png](https://s2.ax1x.com/2019/10/29/Kfl1eI.png)

- 区域分割

区域分割算法中比较有代表性的算法有两种：区域生长和区域分裂合并。

区域生长算法的核心思想是给定子区域一个种子像素，作为生长的起点，然后将种子像素周围邻域中与种子像素有相同或相似性质的像素(可以根据预先设定的规则，比如基于灰度差)合并到种子所在的区域中。

区域分裂合并基本上就是区域生长的逆过程，从整个图像出发，不断分裂得到各个子区域，然后再把前景区域合并，实现目标提取。

- 聚类

聚类是一个应用非常广泛的无监督学习算法，该算法在图像分割领域也有较多的应用。聚类的核心思想就是利用样本的相似性，把相似的像素点聚合成同一个子区域。

- 边缘分割

这是图像分割中较为成熟，而且较为常用的一类算法。边缘分割主要利用图像在边缘处灰度级会发生突变来对图像进行分割。常用的方法是利用差分求图像梯度，而在物体边缘处，梯度幅值会较大，所以可以利用梯度阈值进行分割，得到物体的边缘。对于阶跃状边缘，其位置对应一阶导数的极值点，对应二阶导数的过零点(零交叉点)。因此常用微分算子进行边缘检测。常用的一阶微分算子有Roberts算子、Prewitt算子和Sobel算子，二阶微分算子有Laplace算子和Kirsh算子等。由于边缘和噪声都是灰度不连续点，在频域均为高频分量，直接采用微分运算难以克服噪声的影响。因此用微分算子检测边缘前要对图像进行平滑滤波。LoG算子和Canny算子是具有平滑功能的二阶和一阶微分算子，边缘检测效果较好，因此Canny算子也是应用较多的一种边缘分割算法。

![Kfl8TP.png](https://s2.ax1x.com/2019/10/29/Kfl8TP.png)

- 直方图

与前面提到的算法不同，直方图图像分割算法利用统计信息对图像进行分割。通过统计图像中的像素，得到图像的灰度直方图，然后在直方图的波峰和波谷是用于定位图像中的簇。

- 水平集

水平集方法最初由Osher和Sethian提出，目的是用于界面追踪。在90年代末期被广泛应用在各种图像领域。这一方法能够在隐式有效的应对曲线/曲面演化问题。基本思想是用一个符号函数表示演化中的轮廓（曲线或曲面），其中符号函数的零水平面对应于实际的轮廓。这样对应于轮廓运动方程，可以容易的导出隐式曲线/曲面的相似曲线流，当应用在零水平面上将会反映轮廓自身的演化。水平集方法具有许多优点：它是隐式的，参数自由的，提供了一种估计演化中的几何性质的直接方法，能够改变拓扑结构并且是本质的。

![KflUSg.png](https://s2.ax1x.com/2019/10/29/KflUSg.png)

语义分割和非语义分割的共同之处都是要分割出图像中物体的边缘，但是二者也有本质的区别，用通俗的话介绍就是非语义分割只想提取物体的边缘，但是不关注目标的类别。而语义分割不仅要提取到边缘像素级别，还要知道这个目标是什么。因此，非语义分割是一种图像基础处理技术，而语义分割是一种机器视觉技术，难度也更大一些，目前比较成熟且应用广泛的语义分割算法有以下几种：

- Grab cut
- Mask R-CNN
- U-Net
- FCN
- SegNet

由于篇幅有限，所以在这里就展开介绍语义分割，后期有时间会单独对某几个算法进行详细解析，本文主要介绍非语义分割算法，本文就以2015年UCLA提出的一种新型、高效的图像分割算法--相位拉伸变换为例，详细介绍一下，并从头到尾实现一遍。

## 相位拉伸变换

相位拉伸变换(Phase Stretch Transform, PST)，是UCLA JalaliLab于2015年提出的一种新型图像分割算法[[Edge Detection in Digital Images Using Dispersive Phase Stretch Transform](http://downloads.hindawi.com/journals/ijbi/2015/687819.pdf)]，该算法主要有两个显著优点：

- 速度快
- 精度高
- 思想简单
- 实现容易

PST算法中，首先使用定位核对原始图像进行平滑，然后通过非线性频率相关（离散）相位操作，称为相位拉伸变换(PST)。 PST将2D相位函数应用于频域中的图像。施加到图像的相位量取决于频率;也就是说，较高的相位量被应用于图像的较高频率特征。由于图像边缘包含更高频率的特征，因此PST通过将更多相位应用于更高频率的特征来强调图像中的边缘信息。可以通过对PST输出相位图像进行阈值处理来提取图像边缘。在阈值处理之后，通过形态学操作进一步处理二值图像以找到图像边缘。思想主要包含三个步骤：

![KflwOs.png](https://s2.ax1x.com/2019/10/29/KflwOs.png)

- 非线性相位离散化
- 阈值化处理
- 形态学运算

下面来详细介绍一下。

相位拉伸变换，核心就是一个公式，

$$A[n, m]=\angle(IFFT2(\tilde{K}[p, q]\cdot\tilde{L}[p, q]\cdot FFT2(B[n, m]))) \tag{1}$$

其中 $B[n, m]$ 为输入图像， $m,n$ 为图像维数， $A[n, m]$ 为输出图像， $\angle$ 为角运算， $FFT2$ 为快速傅里叶变换， $IFFT2$ 为逆快速傅里叶变换，$p$和 $q$ 是二维频率变量， $\tilde{L}[p, q]$ 为局部频率响应核，通俗的讲，就是一个用于图像平滑、去噪的滤波核，论文中没有给出，可以使用一些用于图像平滑的滤波核代替， $\tilde{K}[p, q]$ 为相位核，其中，

$$\tilde{K}[p, q]=e^{j\cdot\varphi[p,q]} $$

$$\varphi[p,q]=\varphi_{polar}[r,\theta] \\=\varphi_{polar}[r]\\=S\cdot\frac{W\cdot r \cdot tan^{-1}(W \cdot r)-(1/2)\cdot ln(1+(W \cdot r)^2)}{W \cdot r_{max} \cdot tan^{-1}(W \cdot r_{max})-(1/2)ln(1+W \cdot r_{max})^2} \tag{2} $$

$r=\sqrt{p^2+q^2}$， $\theta=tan^{-1}(q/p)$， $S$ 和 $W$ 是施加到图像相位的强度和扭曲，是影响图像分割效果的两个重要参数。

## 编程实践

PST算法中最核心的就是公式(1)，编程实现可以一步一步来实现公式中的每个模块。

首先导入需要的模块，

```python
import os 
import numpy as np
import mahotas as mh
import matplotlib.pylab as plt
import cv2
```

定义全局变量，

```python
L = 0.5 
S = 0.48 
W= 12.14
Threshold_min = -1
Threshold_max = 0.0019
FLAG = 1
```

计算公式中的核心参数， ![r，\theta](https://www.zhihu.com/equation?tex=r%EF%BC%8C%5Ctheta)r，\theta ,

```python
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho
```

生成变量 $p$ 和 $q$ ,

```python
x = np.linspace(-L, L, img.shape[0])
y = np.linspace(-L, L, img.shape[1])
X, Y = np.meshgrid(x, y)
p, q = X.T, y.T
theta, rho = cart2pol(p, q)
```

接下来对公式(1)从右至左依次实现，

对输入图像进行快速傅里叶变换,

```python
orig = np.fft.fft2(img)
```

实现 $\tilde{L}[p, q]$，

```python
expo = np.fft.fftshift(np.exp(-np.power((np.divide(rho, math.sqrt((LPF ** 2) / np.log(2)))), 2)))
```

对图像进行平滑处理，

```python
orig_filtered = np.real(np.fft.ifft2((np.multiply(orig, expo))))
```

实现相位核，

```python
PST_Kernel_1 = np.multiply(np.dot(rho, W), np.arctan(np.dot(rho, W))) - 0.5 * np.log(1 + np.power(np.dot(rho, W), 2))
PST_Kernel = PST_Kernel_1 / np.max(PST_Kernel_1) * S
```

将前面实现的部分与相位核做乘积，

```python
temp = np.multiply(np.fft.fftshift(np.exp(-1j * PST_Kernel)), np.fft.fft2(orig_filtered))
```

对图像进行逆快速傅里叶变换，

```python
temp = np.multiply(np.fft.fftshift(np.exp(-1j * PST_Kernel)), np.fft.fft2(Image_orig_filtered))
orig_filtered_PST = np.fft.ifft2(temp)
```

进行角运算，得到变换图像的相位，

```python
PHI_features = np.angle(Image_orig_filtered_PST)
```

对图像进行阈值化处理，

```python
features = np.zeros((PHI_features.shape[0], PHI_features.shape[1]))
features[PHI_features > Threshold_max] = 1 
features[PHI_features < Threshold_min] = 1  
features[I < (np.amax(I) / 20)] = 0
```

应用二进制形态学操作来清除转换后的图像,

```python
out = features
out = mh.thin(out, 1)
out = mh.bwperim(out, 4)
out = mh.thin(out, 1)
out = mh.erode(out, np.ones((1, 1)))
```

到这里就完成了相位拉伸变换的核心部分，

```python
def phase_stretch_transform(img, LPF, S, W, threshold_min, threshold_max, flag):
    L = 0.5
    x = np.linspace(-L, L, img.shape[0])
    y = np.linspace(-L, L, img.shape[1])
    [X1, Y1] = (np.meshgrid(x, y))
    X = X1.T
    Y = Y1.T
    theta, rho = cart2pol(X, Y)
    orig = ((np.fft.fft2(img)))
    expo = np.fft.fftshift(np.exp(-np.power((np.divide(rho, math.sqrt((LPF ** 2) / np.log(2)))), 2)))
    orig_filtered = np.real(np.fft.ifft2((np.multiply(orig, expo))))
    PST_Kernel_1 = np.multiply(np.dot(rho, W), np.arctan(np.dot(rho, W))) - 0.5 * np.log(
        1 + np.power(np.dot(rho, W), 2))
    PST_Kernel = PST_Kernel_1 / np.max(PST_Kernel_1) * S
    temp = np.multiply(np.fft.fftshift(np.exp(-1j * PST_Kernel)), np.fft.fft2(orig_filtered))
    orig_filtered_PST = np.fft.ifft2(temp)
    PHI_features = np.angle(orig_filtered_PST)
    if flag == 0:
        out = PHI_features
    else:
        features = np.zeros((PHI_features.shape[0], PHI_features.shape[1]))
        features[PHI_features > threshold_max] = 1
        features[PHI_features < threshold_min] = 1
        features[img < (np.amax(img) / 20)] = 0

        out = features
        out = mh.thin(out, 1)
        out = mh.bwperim(out, 4)
        out = mh.thin(out, 1)
        out = mh.erode(out, np.ones((1, 1)))
    return out, PST_Kernel
```

下面完成调用部分的功能，

首先读取函数并把图像转化为灰度图，

```python
Image_orig = mh.imread("./cameraman.tif")
if Image_orig.ndim == 3:
    Image_orig_grey = mh.colors.rgb2grey(Image_orig) 
else: 
    Image_orig_grey = Image_orig
```

调用前面的函数，对图像进行相位拉伸变换，

```python
edge, kernel = phase_stretch_transform(Image_orig_grey, LPF, Phase_strength, Warp_strength, Threshold_min, Threshold_max, Morph_flag)
```

显示图像，

```python
Overlay = mh.overlay(Image_orig_grey, edge)
edge = edge.astype(np.uint8)*255
plt.imshow(Edge)
plt.show()
```

主函数的完整内容为，

```python
def main():
    Image_orig = mh.imread("./cameraman.tif")
    if Image_orig.ndim == 3:
       Image_orig_grey = mh.colors.rgb2grey(Image_orig)
    else:
       Image_orig_grey = Image_orig
    edge, kernel = phase_stretch_transform(Image_orig_grey, LPF, S, W, Threshold_min,
                                           Threshold_max, FLAG)
    Overlay = mh.overlay(Image_orig_grey, Edge)
    Edge = Edge.astype(np.uint8)*255
    plt.imshow(Edge)
    plt.show()
```

![Kf16HI.png](https://s2.ax1x.com/2019/10/29/Kf16HI.png)

------

## 往期回顾

[Jackpop：【动手学计算机视觉】第一讲：图像预处理之图像去噪](https://zhuanlan.zhihu.com/p/57521026)

[Jackpop：【动手学计算机视觉】第二讲：图像预处理之图像增强](https://zhuanlan.zhihu.com/p/57537622)

!img](https://pic2.zhimg.com/v2-4da90f547b115c1a398b3cc516eada3d_b.png)

> 感兴趣的可以关注一下，也可以关注公众号"平凡而诗意"，我在公众号共享了一些资源和学习资料，关注后回复相应关键字可以获取。