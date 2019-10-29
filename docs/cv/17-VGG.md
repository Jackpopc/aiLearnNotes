---
title: 【动手学计算机视觉】第十七讲：卷积神经网络之VGG
---

# 前言

[![KyYVj1.md.png](https://s2.ax1x.com/2019/10/27/KyYVj1.md.png)](https://imgchr.com/i/KyYVj1)

2014年对于计算机视觉领域是一个丰收的一年，在这一年的ImageNet图像识别挑战赛(ILSVRC,ImageNet Large Scale Visual Recognition Challenge)中出现了两个经典、响至深的卷积神经网络模型，其中第一名是GoogLeNet、第二名是VGG，都可以称得上是深度计算机视觉发展过程中的经典之作。

虽然在名次上GoogLeNet盖过了VGG，但是在可迁移性方面GoogLeNet对比于VGG却有很大的差距，而且在模型构建思想方面对比于它之前的AlexNet、LeNet做出了很大的改进，因此，VGG后来常作为后续卷积神经网络模型的基础模块，用于特征提取。直到5年后的今天，依然可以在很多新颖的CNN模型中可以见到VGG的身影，本文就来详细介绍一下这个经典的卷积神经网络模型。		

# VGG模型

![config](https://s2.ax1x.com/2019/10/27/KytAIS.png)

VGG(VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION)，是由牛津大学的研究者提出，它的名称也是以作者所在实验室而命名(Visual Geometry Group)。

前一篇文章介绍了经典的AlexNet，虽然它在识别效果方面非常令人惊艳，但是这些都是建立在对超参数进行大量的调整的基础上，而它并没有提出一种明确的模型设计规则以便指导后续的新网络模型设计，这也限制了它的迁移能力。因此，虽然它很知名，但是在近几年的模型基础框架却很少出现AlexNet的身影，反观VGG则成为了很多新模型基础框架的必选项之一，这也是它相对于AlexNet的优势之一：<b>VGG提出用基础块代替网络层的思想，这使得它在构建深度网络模型时可以重复使用这些基础块。</b>

正如前面所说，VGG使用了<b>块</b>代替<b>层</b>的思想，具体的来说，它提出了构建基础的<b>卷积块</b>和<b>全连接块</b>来替代<b>卷积层</b>和<b>全连接层</b>，而这里的<b>块</b>是由多个<b>输出通道相同</b>的层组成。

VGG和AlexNet指代单一的模型不同，VGG其实包含多个不同的模型，从上图可以看出，它主要包括下列模型，

- VGG-11
- VGG-13
- VGG-16
- VGG-19

其中，后面的数字11、13、16、19是网络层数。

从图中可以看出，VGG的特点是每个<b>卷积块</b>(由1个或多个卷积层组成)后面跟随一个最大池化层，整体架构和AlexNet非常类似，主要区别就是把层替换成了块。

从图中红框标记可以看出，每个卷积块中输出通道数相同，另外从横向维度来看，不同模型在相同卷积块中输出通道也相同。

下面就以比较常用的VGG-16这个模型为例来介绍一下VGG的模型架构。

VGG-16是由<b>5个卷积块</b>和<b>3个全连接层</b>共8部分组成(回想一下，AlexNet也是由8个部分组成，只不过AlexNet是由5个卷积层和3个全连接层组成)，下面详细介绍每一个部门的详细情况。

<b>注意：</b>前两篇文章我们在搭建LeNet和AlexNet时会发现，不同层的卷积核、步长均有差别，这也是迁移过程中比较困难的一点，而在VGG中就没有这样的困扰，VGG卷积块中统一采用的是<b>3\*3</b>的卷积核，卷积层的步长均为<b>1</b>，而在池化层窗口大小统一采用<b>2\*2</b>，步长为<b>2</b>。因为每个卷积层、池化层窗口大小、步长都是确定的，因此要搭建VGG我们只需要关注每一层输入输出的通道数即可。

<b>卷积块1</b>

包含<b>2</b>个卷积层，输入是<b>224\*224\*3</b>的图像，输入通道数为<b>3</b>，输出通道数为<b>64</b>。

<b>卷积块2</b>

包含<b>2</b>个卷积层，输入是上一个卷积块的输出，输入通道数为<b>64</b>，输出通道数为<b>128</b>。

<b>卷积块3</b>

包含<b>3</b>个卷积层，输入是上一个卷积块的输出，输入通道数为<b>128</b>，输出通道数为<b>256</b>。

<b>卷积块4</b>

包含<b>3</b>个卷积层，输入是上一个卷积块的输出，输入通道数为<b>256</b>，输出通道数为<b>512</b>。

<b>卷积块5</b>

包含<b>3</b>个卷积层，输入是上一个卷积块的输出，输入通道数为<b>512</b>，输出通道数为<b>512</b>。

<b>全连接层1</b>

输入为上一层的输出，输入通道数为前一卷积块输出reshape成一维的长度,输出通道数为<b>4096</b>。

<b>全连接层2</b>

输入为上一层的输出，输入通道数为<b>4096</b>,输出通道数为<b>4096</b>。

<b>全连接层3</b>

输入为上一层的输出，输入通道数为<b>4096</b>,输出通道数为<b>1000</b>。

<b>激活函数</b>

VGG中每层使用的激活函数为<b>ReLU</b>激活函数。

由于VGG非常经典，所以，网络上有关于VGG-16、VGG-19预训练的权重，为了为了展示一下每一层的架构，读取VGG-16预训练权重看一下，

```python
import numpy as np

path = "vgg16.npy"
layers = ["conv1_1", "conv1_2",
          "conv2_1", "conv2_2",
          "conv3_1", "conv3_2", "conv3_3",
          "conv4_1", "conv4_2", "conv4_3",
          "conv5_1", "conv5_2", "conv5_3",
          "fc6", "fc7", "fc8"]

data_dict = np.load(path, encoding='latin1').item()

for layer in layers:
    print(data_dict[layer][0].shape)
    
# 输出
(3, 3, 3, 64)
(3, 3, 64, 64)
(3, 3, 64, 128)
(3, 3, 128, 128)
(3, 3, 128, 256)
(3, 3, 256, 256)
(3, 3, 256, 256)
(3, 3, 256, 512)
(3, 3, 512, 512)
(3, 3, 512, 512)
(3, 3, 512, 512)
(3, 3, 512, 512)
(3, 3, 512, 512)
(25088, 4096)
(4096, 4096)
(4096, 1000)
```

网络共16层，卷积层部分为<b>1\*4</b>维的，其中从前到后分别是<b>卷积核高度</b>、<b>卷积核宽度</b>、<b>输入数据通道数</b>、<b>输出数据通道数</b>。

到此为止，应该已经了解了VGG的模型结构，下面就开始使用tensorflow编程实现一下 VGG。

# 编程实践

因为 VGG非常经典，所以网络上有VGG的预训练权重，我们可以直接读取预训练的权重去搭建模型，这样就可以忽略对输入和输出通道数的感知，要简单很多，但是为了更加清楚的理解网络模型，在这里还是从最基本的部分开始搭建，自己初始化权重和偏差，这样能够更加清楚每层输入和输出的结构。

<b>卷积块</b>

经过前面的介绍应该了解，VGG的主要特点就在于卷积块的使用，因此，我们首先来完成卷积块部分的编写。在完成一段代码的编写之前，我们应该首先弄明白两点：输入和输出。

输出当然很明确，就是经过每个卷积块(多个卷积层)卷积、激活后的tensor，我们要明确的就是应该输入哪些参数？

最重要的3个输入：<b>要进行运算的tensor</b>、<b>每个卷积块内卷积层的个数</b>、<b>输出通道数</b>。

当然，我们为了更加规范的搭建模型，也需要对每一层规定一个命名空间，这样还需要输入每一层的名称。至于<b>输入通道数</b>，我们可以通过tensorflow的get_shape函数获取，

```python
def conv_block(self, X, num_layers, block_index, num_channels):
    in_channels = int(X.get_shape()[-1])
    for i in range(num_layers):
        name = "conv{}_{}".format(block_index, i)
        with tf.variable_scope(name) as scope:
            weight = tf.get_variable("weight", [3, 3, in_channels, num_channels])
            bias = tf.get_variable("bias", [num_channels])
        conv = tf.nn.conv2d(X, weight, strides=[1, 1, 1, 1], padding="SAME")
        X = tf.nn.relu(tf.nn.bias_add(conv, bias))
        in_channels = num_channels
        print(X.get_shape())
    return X
```

从代码中可以看出，有几个参数是固定的：

- 卷积窗口大小
- 步长
- 填充方式
- 激活函数

到此为止，我们就完成了VGG最核心一部分的搭建。

<b>池化层</b>

之前看过前两篇关于AlexNet、LeNet的同学应该记得，池化层有两个重要的参数：<b>窗口大小</b>、<b>步长</b>。由于在VGG中这两个超参数是固定的，因此，不用再作为函数的入参，直接写在代码中即可。

```python
def max_pool(self, X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
```

<b>全连接层</b>

至于全连接层，和前面介绍的两个模型没有什么区别，我们只需要知道输出通道数即可，每一层的输出为上一层的输出，

```python
def full_connect_layer(self, X, out_filters, name):
    in_filters = X.get_shape()[-1]
    with tf.variable_scope(name) as scope:
        w_fc = tf.get_variable("weight", shape=[in_filters, out_filters])
        b_fc = tf.get_variable("bias", shape=[out_filters], trainable=True)
    fc = tf.nn.xw_plus_b(X, w_fc, b_fc)
    return tf.nn.relu(fc)
```

由于不同网络模型之前主要的不同之处就在于模型的结构，至于训练和验证过程中需要的准确率、损失函数、优化函数等都大同小异，在前两篇文章中已经实现了训练和验证部分，所以这里就不再赘述。在本文里，我使用numpy生成一个随机的测试集测试一下网络模型是否搭建成功即可。

<b>测试</b>

首先使用numpy生成符合正态分布的随机数，形状为(5, 224, 224, 3)，5为批量数据的大小，244为输入图像的尺寸，3为输入图像的通道数，设定输出类别数为1000，

```python
def main():
    X = np.random.normal(size=(5, 224, 224, 3))
    images = tf.placeholder("float", [5, 224, 224, 3])
    vgg = VGG(1000)
    writer = tf.summary.FileWriter("logs")
    with tf.Session() as sess:
        model = vgg.create(images)
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        prob = sess.run(model, feed_dict={images: X})
        print(sess.run(tf.argmax(prob, 1)))

# 输出
(5, 224, 224, 64)
(5, 224, 224, 64)
(5, 112, 112, 128)
(5, 112, 112, 128)
(5, 56, 56, 256)
(5, 56, 56, 256)
(5, 56, 56, 256)
(5, 28, 28, 512)
(5, 28, 28, 512)
(5, 28, 28, 512)
(5, 14, 14, 512)
(5, 14, 14, 512)
(5, 14, 14, 512)
(5, 4096)
(5, 4096)
(5, 1000)
[862 862 862 862 862]
```

可以对比看出，每层网络的尺寸和前面加载的预训练模型是匹配的，下面在看一下tensorboard的结果，

```shell
$ tensorboard --logdir="logs"
```

结果，

[![KyY8gA.md.gif](https://s2.ax1x.com/2019/10/27/KyY8gA.md.gif)](https://imgchr.com/i/KyY8gA)

# 完整代码

完整代码请查看github项目<b>aiLearnNotes</b>，也可以直接访问下面链接，

https://github.com/Jackpopc/aiLearnNotes/blob/master/computer_vision/VGG-16.py