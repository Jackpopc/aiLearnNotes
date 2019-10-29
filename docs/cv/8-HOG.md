---
title: 【动手学计算机视觉】第八讲：传统目标检测之HOG特征
---

## 前言

![erkAC8.png](https://s2.ax1x.com/2019/08/03/erkAC8.png)

如果自称为计算机视觉工程师，没有听说过前文提到的尺度不变特征变换(SIFT)，可以理解，但是如果没有听说过方向梯度直方图(Histogram of oriented gradient，HOG)，就有一些令人诧异了。这项技术是有发过国家计算机技术和控制研究所(INRIA)的两位研究院Navneet Dalal和Bill Triggs在2005年CVPR上首先发表提出(那时的CVPR含金量还是很高的)。原文Histograms of oriented gradients for human detection截止2019年7月10日引用率已经达到26856。

HOG通过计算局部图像提取的方向信息统计值来统计图像的梯度特征，它跟EOH、SIFT及shape contexts有诸多相似之处，但是它有明显的不同之处：HOG特征描述子是在一个网格秘籍、大小统一的细胞单元上进行计算，而且为了提高性能，它还采用了局部对比度归一化思想。它的出现，使得目标检测技术在静态图像的人物检测、车辆检测等方向得到大量应用。

在传统目标检测中，HOG可以称得上是经典中的经典，它的HOG+SVM+归一化思想对后面的研究产生深远的影响，包括后面要讲到的神作DPM，可以说，HOG的出现，奠定了2005之后的传统目标检测的基调和方向，下面就来了解一下这个经典之作。

## 方向梯度直方图

![erkmuj.png](https://s2.ax1x.com/2019/08/03/erkmuj.png)

HOG特征的算法可以用一下几个部分概括，

- 梯度计算
- 单元划分
- 区块选择
- 区间归一化
- SVM分类器

下面分别来详细阐述一下。

![eR8K9x.png](https://s2.ax1x.com/2019/08/05/eR8K9x.png)

### 梯度计算

由于后面要进行归一化处理，因此在HOG中不需要像其他算法那样需要进行预处理，因此，第一步就成了梯度计算。为什么选择梯度特征？因为在目标边缘处灰度变化较大，因此，在边缘处灰度的梯度就较为明显，所以，梯度能够更好的表征目标的特征。

我们都知道在数学中计算梯度需要进行微分求导，但是数字图像是离散的，因此无法直接求导，可以利用一阶差分代替微分求离散图像的梯度大小和梯度方向，计算得到水平方向和垂直方向的梯度分别是，

$$G_{h}(x, y)=f(x+1, y)-f(x-1, y),\forall x, y$$

$$G_{v}(x, y)=f(x, y+1)-f(x, y-1) ,\forall x, y$$

其中$f(x,y)$表示图像在$(x,y)$的像素值1。

可以得到梯度值(梯度强度)和梯度方向分别为,

$$M(x, y)=\sqrt{G_{h}(x, y)^{2}+G_{v}(x, y)^{2}}$$

$$\theta(x, y)=\arctan \left(G_{h}(x, y) / G_{v}(x, y)\right.$$

### 单元划分

![erkiUP.png](https://s2.ax1x.com/2019/08/03/erkiUP.png)

计算得到梯度的幅值和梯度方向之后，紧接着就是要建立分块直方图，得到图像的梯度大小和梯度方向后根据梯度方向对图像进行投影统计，首先将图像划分成若干个块(Block)，每个块又由若干个细胞单元(cell)组成，细胞单元由更小的单位像素(Pixel)组成，然后在每个细胞单元中对内部的所有像素的梯度方向进行统计。Dalal和Triggs通过测试验证得出，把方向分为9个通道效果最好，因此将180度划分成9个区间，每个区间为20度，如果像素落在某个区间，就将该像素的直方图累加在该区间对应的直方图上面，例如，如果像素的梯度方向在0~20度之间，则在0~20对应的直方图上累加该像素对应的梯度幅值。这样最终每个细胞单元就会得到一个9维的特征向量，特征向量每一维对应的值是累加的梯度幅值。

![erkF4f.png](https://s2.ax1x.com/2019/08/03/erkF4f.png)

### 区块选择

为了应对光照和形变，梯度需要在局部进行归一化。这个局部的区块该怎么选择？常用的有两种，分别是矩形区块(R-HOG)和圆形区块(C-HOG)，前面提供的例子就是矩形区块，一个矩形区块由三个参数表示：每个区块由多少放歌、每个方格有多少像素、每个像素有多少通道。前面已经提到，经过作者验证，每个像素选择9个通道效果最佳。同样，作者对每个方格采用的像素数也进行验证，经过验证每个方格采用3\*3或者6\*6个像素效果较好。

### 区间归一化

每个方格内对像素梯度方向进行统计可以得出一个特征向量，一个区块内有多个方格，也就有多个特征向量，例如前面的示例区块Block内就有4个9维向量。这一步要做的就是对这4个向量进行归一化，Dalal和Triggs采用了四种不同的方式对区块进行归一化，分别是L2-norm、L2-hys、L1-norm、L1-sqrt，用$v$表示未被归一化的向量，以L2-norm为例，归一化后的特征向量为，

$$v=\frac{v}{\sqrt{\|v\|_{2}^{2}+\varepsilon^{2}}}$$

作者通过对比发现，L2-norm、L2-hys、L1-sqrt三种方式所取得的效果是一样的，L1-norm表现相对差一些。

### SVM分类器

最后一步，也是比较关键的一步，就是训练分类器，用SVM对前面提取的图像特征向量进行训练，寻找一个最优超平面作为决策函数，得到目标的训练模型。

## 编程实践

完整代码请查看：

https://github.com/Jackpopc/aiLearnNotes/blob/master/computer_vision/HOG.py

HOG是一个优秀的特征提取算法，因此本文就仅介绍并实现特征提取算法部分，后面的训练分类器和目标检测偏重于机器学习内容，在这里就不多赘述。

HOG算法非常经典，因此，很多成熟的第三方库都已经集成了这个算法，例如比较知名的计算机视觉库OpenCV，对于HOG特征提取比较简单的方式就是直接调用OpenCV库，具体代码如下，

```python
import cv2
hog = cv2.HOGDescriptor()
img = cv2.imread("../data/2007_000129.jpg", cv2.IMREAD_GRAYSCALE)
des = hog.compute(img)
```

为了更好的理解HOG算法，本文就跟随文章的思路来重新实现一遍算法。

**第一步：计算梯度方向和梯度幅值**

这里用Sobel算子来计算水平和垂直方向的差分，然后用对梯度大小**加权求和**的方式来计算统计时使用的梯度幅值，

```python
def compute_image_gradient(img):
    x_values = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    y_values = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.addWeighted(x_values, 0.5, y_values, 0.5, 0)
    angle = cv2.phase(x_values, y_values, angleInDegrees=True)
    return magnitude, angle
```

**第二步：统计细胞单元的梯度方向**

指定细胞单元尺寸和角度单元，然后对用直方图统计一个细胞单元内的梯度方向，如果梯度角度落在一个区间内，则把该像素的幅值加权到和角度较近的一个角度区间内，

```python
def compute_cell_gradient(cell_magnitude, cell_angle, bin_size, unit):
    centers = [0] * bin_size
    # 遍历细胞单元，统计梯度方向
    for i in range(cell_magnitude.shape[0]):
        for j in range(cell_magnitude.shape[1]):
            strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]
            min_angle, max_angle, mod = choose_bins(gradient_angle, unit, bin_size)
            # 根据角度的相近程度分别对邻近的两个区间进行加权
            centers[min_angle] += (strength * (1 - (mod / unit)))
            centers[max_angle] += (strength * (mod / unit))
    return centers
```

**第三步：块内归一化**

根据HOG原文的思想可以知道，图像内分块，块内分细胞单元，然后对细胞单元进行统计。一个块由多个细胞单元组成，统计了每个细胞单元的梯度特征之后需要对这几个向量进行归一化，

```python
def normalized(cell_gradient_vector):
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                # 归一化
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    return hog_vector
```

**第四步：可视化**

为了直观的看出特征提取的效果，对下图进行特征提取并且可视化，

![erkZvQ.png](https://s2.ax1x.com/2019/08/03/erkZvQ.png)

可视化的方法是在每个像素上用线段画出梯度的方向和大小，用线段的长度来表示梯度大小，

```python
def visual(cell_gradient, height, width, cell_size, unit):
    feature_image = np.zeros([height, width])
    cell_width = cell_size / 2
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(feature_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_gap
    return feature_image
```

提取的特征图为，图中白色的线段即为提取的特征，

![erkVgg.png](https://s2.ax1x.com/2019/08/03/erkVgg.png)

完整代码如下，

```python
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread("../data/2007_000129.jpg", cv2.IMREAD_GRAYSCALE)


def compute_image_gradient(img):
    x_values = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    y_values = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = abs(cv2.addWeighted(x_values, 0.5, y_values, 0.5, 0))
    angle = cv2.phase(x_values, y_values, angleInDegrees=True)
    return magnitude, angle


def choose_bins(gradient_angle, unit, bin_size):
    idx = int(gradient_angle / unit)
    mod = gradient_angle % unit
    return idx, (idx + 1) % bin_size, mod


def compute_cell_gradient(cell_magnitude, cell_angle, bin_size, unit):
    centers = [0] * bin_size
    for i in range(cell_magnitude.shape[0]):
        for j in range(cell_magnitude.shape[1]):
            strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]
            min_angle, max_angle, mod = choose_bins(gradient_angle, unit, bin_size)
            print(gradient_angle, unit, min_angle, max_angle)
            centers[min_angle] += (strength * (1 - (mod / unit)))
            centers[max_angle] += (strength * (mod / unit))
    return centers


def normalized(cell_gradient_vector):
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    return hog_vector


def visual(cell_gradient, height, width, cell_size, unit):
    feature_image = np.zeros([height, width])
    cell_width = cell_size / 2
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(feature_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_gap
    return feature_image


def main(img):
    cell_size = 16
    bin_size = 9
    unit = 360 // bin_size
    height, width = img.shape

    magnitude, angle = compute_image_gradient(img)

    cell_gradient_vector = np.zeros((height // cell_size, width // cell_size, bin_size))
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            cell_gradient_vector[i][j] = compute_cell_gradient(cell_magnitude, cell_angle, bin_size, unit)
    hog_vector = normalized(cell_gradient_vector)
    hog_image = visual(cell_gradient_vector, height, width, cell_size, unit)
    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('../data/2007_002293.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("origin", img)
    cv2.waitKey()
    main(img)
```

