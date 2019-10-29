---
title: 【动手学计算机视觉】第七讲：传统目标检测之SIFT特征
---

## **前言**

提到传统目标识别，就不得不提SIFT算法，Scale-invariant feature transform，中文含义就是尺度不变特征变换。此方法由David Lowe于1999年发表于ICCV(International Conference on Computer Vision)，并经过5年的整理和晚上，在2004年发表于IJCV(International journal of computer vision)。由于在此之前的目标检测算法对图片的大小、旋转非常敏感，而SIFT算法是一种基于局部兴趣点的算法，因此不仅对图片大小和旋转不敏感，而且对光照、噪声等影响的抗击能力也非常优秀，因此，该算法在性能和适用范围方面较于之前的算法有着质的改变。这使得该算法对比于之前的算法有着明显的优势，所以，一直以来它都在目标检测和特征提取方向占据着重要的地位，截止2019年6月19日，这篇文章的引用量已经达到51330次(谷歌学术)，受欢迎程度可见一斑，本文就详细介绍一下这篇文章的原理，并一步一步编程实现本算法，让各位对这个算法有更清晰的认识和理解。

## **SIFT**

前面提到，SIFT是一个非常经典而且受欢迎的特征描述算法，因此关于这篇文章的学习资料、文章介绍自然非常多。但是很多文章都相当于把原文翻译一遍，花大量篇幅在讲高斯模糊、尺度空间理论、高斯金字塔等内容，容易让人云里雾里，不知道这种算法到底在讲什么？重点又在哪里？

![Kft5vT.png](https://s2.ax1x.com/2019/10/29/Kft5vT.png)

图1 SIFT算法步骤

其实下载这篇文章之后打开看一下会发现，SIFT的思想并没有想的那么复杂，它主要包含4个步骤：

- **尺度空间极值检测**：通过使用高斯差分函数来计算并搜索所有尺度上的图像位置，用于识别对尺度和方向不变的潜在兴趣点。

![Kftqa9.png](https://s2.ax1x.com/2019/10/29/Kftqa9.png)

- **关键点定位**：通过一个拟合精细的模型在每个候选位置上确定位置和尺度，关键点的选择依赖于它们的稳定程度。

![KftwCt.png](https://s2.ax1x.com/2019/10/29/KftwCt.png)

- **方向匹配**：基于局部图像的梯度方向，为每个关键点位置分配一个或多个方向，后续所有对图像数据的操作都是相对于关键点的方向、尺度和位置进行变换，从而而这些变换提供了不变形。
- **关键点描述**：这个和HOG算法有点类似之处，在每个关键点周围的区域内以选定的比例计算局部图像梯度，这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。

由于它将图像数据转换为相对于局部特征的尺度不变坐标，因此这种方法被称为尺度不变特征变换。

如果对这个算法思路进行简化，它就包括2个部分：

- 特征提取
- 特征描述

## **特征提取**

特征点检测主要分为如下两个部分，

- 候选关键点
- 关键点定位

**候选关键点**

Koenderink（1984）和Lindeberg（1994）已经证明，在各种合理的假设下，高斯函数是唯一可能的尺度空间核。因此，图像的尺度空间被定义为函数，它是由一个可变尺度的高斯核和输入图像生成，  其中高斯核为，  为了有效检测尺度空间中稳定的极点，Lowe于1999年提出在高斯差分函数(DOG)中使用尺度空间极值与图像做卷积，这可以通过由常数乘法因子分隔的两个相邻尺度的差来计算。用公式表示就是，  由于平滑区域临近像素之间变化不大，但是在边、角、点这些特征较丰富的地方变化较大，因此通过DOG比较临近像素可以检测出候选关键点。

**关键点定位**

检测出候选关键点之后，下一步就是通过拟合惊喜的模型来确定位置和尺度。 2002年Brown提出了一种用3D二次函数来你和局部样本点，来确定最大值的插值位置，实验表明，这使得匹配和稳定性得到了实质的改进。 他的具体方法是对函数进行泰勒展开，  上述的展开式，就是所要的拟合函数。 极值点的偏移量为，  如果偏移量在任何一个维度上大于0.5时，则认为插值中心已经偏移到它的邻近点上，所以需要改变当前关键点的位置，同时在新的位置上重复采用插值直到收敛为止。如果超出预先设定的迭代次数或者超出图像的边界，则删除这个点。

## **特征描述**

前面讲了一些有关特征点检测的内容，但是SIFT实质的内容和价值并不在于特征点的检测，而是特征描述思想，这是它的核心所在，特征点描述主要包括如下两点：

- 方向分配
- 局部特征描述

**方向分配**

根据图像的图像，可以为每个关键定指定一个基准方向，可以相对于这个指定方向表示关键点的描述符，从而实现了图像的旋转不变性。 关键点的尺度用于选择尺度最接近的高斯平滑图像，使得计算是以尺度不变的方式执行，对每个图像，分别计算它的梯度幅值和梯度方向，   然后，使用方向直方图统计关键点邻域内的梯度幅值和梯度方向。将0~360度划分成36个区间，每个区间为10度，统计得出的直方图峰值代表关键点的主方向。

**局部特征描述**

通过前面的一系列操作，已经获得每个关键点的**位置、尺度、方向**，接下来要做的就是用已知特征向量把它描述出来，这是图像特征提取的核心部分。为了避免对光照、视角等因素的敏感性，需要特征描述子不仅仅包含关键点，还要包含它的邻域信息。 

![Kft08P.png](https://s2.ax1x.com/2019/10/29/Kft08P.png)

SIFT使用的特征描述子和后面要讲的HOG有很多相似之处。它一检测得到的关键点为中心，选择一个16*16的邻域，然后再把这个邻域再划分为4*4的子区域，然后对梯度方向进行划分成8个区间，这样在每个子区域内疚会得到一个4*4*8=128维的特征向量，向量元素大小为每个梯度方向区间权值。提出得到特征向量后要对邻域的特征向量进行归一化，归一化的方向是计算邻域关键点的主方向，并将邻域旋转至根据主方向旋转至特定方向，这样就使得特征具有旋转不变性。然后再根据邻域内各像素的大小把邻域缩放到指定尺度，进一步使得特征描述子具有尺度不变性。

以上就是SIFT算法的核心部分。

## **编程实践**

本文代码已经放在github，感兴趣的可以自行查看，

https://github.com/jakpopc/aiLearnNotes/blob/master/computer_vision/SIFT.pygithub.com

本文实现SIFT特征检测主要基于以下工具包：

- OpenCV
- numpy

其中OpenCV是一个非常知名且受欢迎的跨平台计算机视觉库，它不仅包含常用的图像读取、显示、颜色变换，还包含一些为人熟知的经典特征检测算法，其中就包括SIFT，所以本文使用OpenCV进行读取和SIFT特征检测。 numpy是一个非常优秀的数值计算库，也常用于图像的处理，这里使用numpy主要用于图像的拼接和显示。

**导入工具包**

```python
import numpy as np
import cv2
```

**图像准备**

首先写一下读取图像的函数，

```python
def load_image(path, gray=True):
    if gray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return cv2.imread(path)
```

然后，生成一副对原图进行变换的图像，用于后面特征匹配，本文选择对图像进行垂直镜像变换，

```python
def transform(origin):
    h, w = origin.shape
    generate_img = np.zeros(origin.shape)
    for i in range(h):
        for j in range(w):
            generate_img[i, w - 1 - j] = origin[i, j]
    return generate_img.astype(np.uint8)
```

显示一下图像变换的结果，

```python
img1 = load_image('2007_002545.jpg')
img2 = transform(img1)
combine = np.hstack((img1, img2))
cv2.imshow("gray", combine)
cv2.waitKey(0)
```

![KftyDg.png](https://s2.ax1x.com/2019/10/29/KftyDg.png)

先用 ***xfeatures2d*** 模块实例化一个sift算子，然后使用 ***detectAndCompute*** 计算关键点和描述子，随后再用 ***drawKeypoints*** 绘出关键点，

```python
# 实例化
sift = cv2.xfeatures2d.SIFT_create()

# 计算关键点和描述子
# 其中kp为关键点keypoints
# des为描述子descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 绘出关键点
# 其中参数分别是源图像、关键点、输出图像、显示颜色
img3 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 255))
img4 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 255))
```

显示出检测的关键点为，

![Kft6bQ.png](https://s2.ax1x.com/2019/10/29/Kft6bQ.png)

关键点已经检测出来，最后一步要做的就是绘出匹配效果，本文用到的是利用 ***FlannBasedMatcher*** 来显示匹配效果， 首先要对 ***FlannBasedMatcher*** 进行参数设计和实例化，然后用 ***knn** 对前面计算的出的特征描述子进行匹配，最后利用 ***drawMatchesKnn*** 显示匹配效果，

```python
# 参数设计和实例化
index_params = dict(algorithm=1, trees=6)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 利用knn计算两个描述子的匹配
matche = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matche))]

# 绘出匹配效果
result = []
for m, n in matche:
    if m.distance < 0.6 * n.distance:
        result.append([m])

img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matche, None, flags=2)
cv2.imshow("MatchResult", img5)
cv2.waitKey(0)
```

检测结果，

![Kft2Us.png](https://s2.ax1x.com/2019/10/29/Kft2Us.png)

完整代码如下，

```python
import numpy as np
import cv2


def load_image(path, gray=False):
    if gray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return cv2.imread(path)


def transform(origin):
    h, w, _ = origin.shape
    generate_img = np.zeros(origin.shape)
    for i in range(h):
        for j in range(w):
            generate_img[i, w - 1 - j] = origin[i, j]
    return generate_img.astype(np.uint8)


def main():
    img1 = load_image('2007_002545.jpg')
    img2 = transform(img1)

    # 实例化
    sift = cv2.xfeatures2d.SIFT_create()

    # 计算关键点和描述子
    # 其中kp为关键点keypoints
    # des为描述子descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 绘出关键点
    # 其中参数分别是源图像、关键点、输出图像、显示颜色
    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 255))
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 255))

    # 参数设计和实例化
    index_params = dict(algorithm=1, trees=6)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 利用knn计算两个描述子的匹配
    matche = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matche))]

    # 绘出匹配效果
    result = []
    for m, n in matche:
        if m.distance < 0.6 * n.distance:
            result.append([m])

    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matche, None, flags=2)
    cv2.imshow("MatchResult", img5)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
```

以上就是SIFT的完整内容。

------

## 往期回顾

[Jackpop：【动手学计算机视觉】第一讲：图像预处理之图像去噪](https://zhuanlan.zhihu.com/p/57521026)

[Jackpop：【动手学计算机视觉】第二讲：图像预处理之图像增强](https://zhuanlan.zhihu.com/p/57537622)

[Jackpop：【动手学计算机视觉】第三讲：图像预处理之图像分割](https://zhuanlan.zhihu.com/p/60847136)

[Jackpop：【动手学计算机视觉】第四讲：图像预处理之图像增广](https://zhuanlan.zhihu.com/p/65367068)

[Jackpop：【动手学计算机视觉】第五讲：传统目标检测之特征工程](https://zhuanlan.zhihu.com/p/66166633)

[Jackpop：【动手学计算机视觉】第六讲：传统目标检测之Harris角点检测](https://zhuanlan.zhihu.com/p/67770305)