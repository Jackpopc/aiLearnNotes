---
title: 【动手学计算机视觉】第一讲：图像预处理之图像去噪
---

## **前言**

很多人想入门AI，可是AI包含很多方向，我建议首先应该明确的选择一个方向，然后有目标、有针对的去学习。

计算机视觉作为目前AI领域研究较多、商业应用较为成功的一个方向，这几年也是非常火热，无论是学术界还是企业界，学术界有CVPR、ICCV、ECCV等顶刊，企业界对计算机视觉领域的人口需求也非常的大，因此，我从计算机视觉这个方向开始着手AI教程。

## **介绍**

最近几年计算机视觉非常火，也出现了很多成熟的卷积神经网络模型，比如R-CNN系列、SSD、YOLO系列，而且，这些模型在github上也有很多不错的开源代码，所以，很多入门计算机视觉的人会早早的克隆下开源代码、利用tensorflow或pytorch搭建计算机视觉平台进行调试。

我个人不推崇这种方式，我更推崇对图像底层的技术有一些了解，比如图像去噪、图像分割等技术，这有几点好处：

- 对图像内部的结构有更清晰的认识
- 这些技术可以用于计算机视觉预处理或后处理，能够有助于提高计算机视觉模型精度

第一讲，我从图像去噪开始说起，图像去噪是指减少图像中造成的过程。现实中的图像会受到各种因素的影响而含有一定的噪声，噪声主要有以下几类：

- 椒盐噪声
- 加性噪声
- 乘性噪声
- 高斯噪声

图像去噪的方法有很多种，其中均值滤波、中值滤波等比较基础且成熟，还有一些基于数学中偏微分方程的去噪方法，此外，还有基于频域的小波去噪方法。均值滤波、中值滤波这些基础的去噪算法以其快速、稳定等特性，在项目中非常受欢迎，在很多成熟的软件或者工具包中也集成了这些算法，下面，我们就来一步一步实现以下。

## **编程实践**

```shell
完整代码地址：
https://github.com/jakpopc/aiLearnNotes/blob/master/computer_vision/image_denoising.py
requirement:scikit-image/opencv/numpy
```

首先读取图像，图像来自于voc2007:

```python
img = cv2.imread("../data/2007_001458.jpg")
cv2.imshow("origin_img", img)
cv2.waitKey()
```

![KfKcfe.png](https://s2.ax1x.com/2019/10/29/KfKcfe.png)

生成噪声图像，就是在原来图像上加上一些分布不规律的像素值，可以自己用随机数去制造噪声，在这里，就用Python第三方库scikit-image的random_noise添加噪声：

***方法1：***

```python
noise_img = skimage.util.random_noise(img, mode="gaussian")
```

> mode是可选参数：分别有'gaussian'、'localvar'、'salt'、'pepper'、's&p'、'speckle'，可以选择添加不同的噪声类型。

***方法2：***

也可以自己生成噪声，与原图像进行加和得到噪声图像：

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

![KfKTk8.png](https://s2.ax1x.com/2019/10/29/KfKTk8.png)

最后是图像去噪，图像去噪的算法有很多，有基于偏微分热传导方程的，也有基于滤波的，其中基于滤波的以其速度快、算法成熟，在很多工具包中都有实现，所以使用也就较多，常用的滤波去噪算法有以下几种：

- 中值滤波
- 均值滤波
- 高斯滤波

滤波的思想和这两年在计算机视觉中用的较多的卷积思想类似，都涉及窗口运算，只是卷积是用一个卷积核和图像中对应位置做卷积运算，而滤波是在窗口内做相应的操作，

![KfKLlj.png](https://s2.ax1x.com/2019/10/29/KfKLlj.png)

以均值滤波为例，

对图像中每个像素的像素值进行重新计算，假设窗口大小ksize=3，图像中棕色的"5"对应的像素实在3*3的邻域窗口内进行计算，对于均值滤波就是求3*3窗口内所有像素点的平均值，也就是

$$\frac{1+2+3+4+6+7+8+9}{9}=4.4 $$

同理，对于中值滤波就是把窗口内像素按像素值大小排序求中间值，高斯滤波就是对整幅图像进行加权平均的过程，每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到，

下面开始编写去噪部分的代码：

方法1：

可以使用opencv这一类工具进行去噪：

```python
# 中值滤波
denoise = cv2.medianBlur(img, ksize=3)
# 均值滤波
denoise = cv2.fastNlMeansDenoising(img, ksize=3)
# 高斯滤波
denoise = cv2.GaussianBlur(img, ksize=3)
```

方法2：

编程一步一步实现图像去噪，首先是计算窗口邻域内的值，这里以计算中值为例：

```python
def compute_pixel_value(img, i, j, ksize, channel):
    h_begin = max(0, i - ksize // 2)
    h_end = min(img.shape[0], i + ksize // 2)
    w_begin = max(0, j - ksize // 2)
    w_end = min(img.shape[1], j + ksize // 2)
    return np.median(img[h_begin:h_end, w_begin:w_end, channel])
```

然后是去噪部分，对每个像素使用compute_pixel_value函数计算新像素的值：

```python
def denoise(img, ksize):
    output = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j, 0] = compute_pixel_value(img, i, j, ksize, 0)
            output[i, j, 1] = compute_pixel_value(img, i, j, ksize, 1)
            output[i, j, 2] = compute_pixel_value(img, i, j, ksize, 2)
    return output
```

![KfMnAK.png](https://s2.ax1x.com/2019/10/29/KfMnAK.png)

------

> 感兴趣的可以关注公众号"平凡而诗意"，共享了一些机器学习、计算机视觉等方面的教材电子版、文章、资源等。