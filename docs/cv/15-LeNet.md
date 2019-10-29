# 前言

![n6JaHe.png](https://s2.ax1x.com/2019/09/14/n6JaHe.png)

提起卷积神经网络，也许可以避开VGG、GoogleNet，甚至可以忽略AleNet，但是很难不提及LeNet。

LeNet是由2019年图灵奖获得者、深度学习三位顶级大牛之二的Yann LeCun、Yoshua Bengio于1998年提出(Gradient-based learning applied to document recognition)，它也被认为被认为是最早的卷积神经网络模型。但是，由于算力和数据集的限制，卷积神经网络提出之后一直都被传统目标识别算法(特征提取+分类器)所压制。终于在沉寂了14年之后的2012年，AlexNet在ImageNet挑战赛上一骑绝尘，使得卷积神经网络又一次成为了研究的热点。

近几年入门计算机视觉的同学大多数都是从AlexNet甚至更新的网络模型入手，了解比较多的就是R-CNN系列和YOLO系列，在很多知名的课程中对LeNet的介绍也是非常浅显或者没有介绍。虽然近几年卷积神经网络模型在LeNet的基础上加入了很多新的单元，在效果方面要明显优于LeNet，但是作为卷积神经网络的基础和源头，它的很多思想对后来的卷积神经网络模型具有很深的影响，因此，我认为了解一下LeNet还是非常有必要的。

本文首先介绍一下LeNet的网络模型，然后使用tensorflow来一步一步实现LeNet。

# LeNet

![n6JwAH.png](https://s2.ax1x.com/2019/09/14/n6JwAH.png)

上图就是LeNet的网络结构，LeNet又被称为LeNet-5，其之所以称为这个名称是由于原始的LeNet是一个5层的卷积神经网络，它主要包括两部分：

- 卷积层
- 全连接层

其中卷积层数为2，全连接层数为3。

<b>这里需要注意一下</b>，之前在介绍卷积、池化时特意提到，在网络层计数中池化和卷积往往是被算作一层的，虽然池化层也被称为"层"，但是它不是一个独立的运算，往往都是紧跟着卷积层使用，因此它不单独计数。在LeNet中也是这样，卷积层块其实是包括两个单元：卷积层与池化层。

在网络模型的搭建过程中，我们关注的除了网络层的结构，还需要关注一些超参数的设定，例如，卷积层中使用卷积核的大小、池化层的步幅等，下面就来介绍一下LeNet详细的网络结构和参数。

<b>第一层：卷积层</b>

卷积核大小为5*5，输入通道数根据图像而定，例如灰度图像为单通道，那么通道数为1，彩色图像为三通道，那么通道数为3。虽然输入通道数是一个变量，但是输出通道数是固定的为6。

池化层中窗口大小为2*2，步幅为2。

<b>第二层：卷积层</b>

卷积核大小为5*5，输入通道数即为上一层的输出通道数6，输出通道数为16。

池化层和第一层相同，窗口大小为2*2，步幅为2。

<b>第三层：全连接层</b>

全连接层顾名思义，就是把卷积层的输出进行展开，变为一个二维的矩阵(第一维是批量样本数，第二位是前一层输出的特征展开后的向量)，输入大小为上一层的输出16，输出大小为120。

<b>第四层：全连接层</b>

输入大小为120，输出大小为84。

<b>第五层：全连接层</b>

输入大小为84，输出大小为类别个数，这个根据不同任务而定，假如是二分类问题，那么输出就是2，对于手写字识别是一个10分类问题，那么输出就是10。

<b>激活函数</b>

前面文章中详细的介绍了激活函数的作用和使用方法，本文就不再赘述。激活函数有很多，例如Sigmoid、relu、双曲正切等，在LeNet中选取的激活函数为Sigmoid。

# 模型构建

![n6J6jf.png](https://s2.ax1x.com/2019/09/14/n6J6jf.png)

如果已经了解一个卷积神经网络模型的结构，知道它有哪些层、每一层长什么样，那样借助目前成熟的机器学习平台是非常容易的，例如tensorflow、pytorch、mxnet、caffe这些都是高度集成的深度学习框架，虽然在强化学习、图神经网络中表现一般，但是在卷积神经网络方面还是很不错的。

我绘制了模型构建的过程，详细的可以看一下上图，很多刚入门的同学会把tensorflow使用、网络搭建看成已经非常困难的事情，其实理清楚之后发现并没有那么复杂，它主要包括如下几个部分：

- 数据输入
- 网络模型
- 训练预测

其中，重点之处就在于网络模型的搭建，需要逐层的去搭建一个卷积神经网络，复杂程度因不同的模型而异。训练测试过程相对简单一些，可以通过交叉熵、均方差等构建损失函数，然后使用深度学习框架自带的优化函数进行优化即可，代码量非常少。

LeNet、AlexNet、VGG、ResNet等，各种卷积神经网络模型主要的区别之处就在于网络模型，但是网络搭建的过程是相同的，均是通过上述流程进行搭建，因此，本文单独用一块内容介绍模型搭建的过程，后续内容不再介绍网络模型的搭建，会直接使用tensorflow进行编程实践。

# 编程实践

<b>完整代码</b>请查看github项目： [aiLearnNotes](https://github.com/Jackpopc/aiLearnNotes/blob/master/computer_vision/LeNet.py)

首先需要说明一下，后续的内容中涉及网络模型搭建的均会选择tensorflow进行编写。虽然近几年pytorch的势头非常迅猛，关于tensorflow的批评之声不绝于耳，但是我一向认为，灵活性和易用性总是成反比的，tensorflow虽然相对复杂，但是它的灵活性非常强，而且支持强大的可视化tensorboard，虽然pytorch也可以借助tensorboard实现可视化，但是这样让我觉得有一些"不伦不类"的感觉，我更加倾向于一体化的框架。此外，有很多同学认为Gluon、keras非常好用，的确，这些在tensorflow、mxnet之上进一步封装的高级深度学习框架非常易用，很多参数甚至不需要开发者去定义，但是正是因为这样，它们已经强行的预先定义在框架里了，可想而知，它的灵活性是非常差的。因此，综合灵活性、一体化、丰富性等方面的考虑，本系列会采用tensorflow进行编程实践。

其次，需要说明的是本系列重点关注的是网络模型，因此，关于数据方面会采用MNIST进行实践。MNIST是一个成熟的手写字数据集，它提供了易用的接口，方便读取和处理。

在使用tensorflow接口读取MNIST时，如果本地有数据，它会从本地加载，否则它会从官网下载数据，如果由于代理或者网速限制的原因自动下载数据失败，可以手动从官网下载数据放在MNIST目录下，数据包括4个文件，分别是：

- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz

它们分别是训练数据集和标签，测试数据集和标签。

可能会有人有疑问，手写体识别不是图像吗？为什么是gz的压缩包？因为作者对手写体进行了序列化处理，方便读取，数据原本是衣服单通道28*28的灰度图像，处理后是784的向量，我们可以通过一段代码对它可视化一下，

```python
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST", one_hot=True)
for i in range(12):
    plt.subplot(3, 4, i+1)
    img = mnist.train.images[i + 1]
    img = img.reshape(28, 28)
    plt.imshow(img)
plt.show()
```

通过读取训练集中的12副图像，然后把它修改成28*28的图像，显示之后会发现和我们常见的图像一样，

![n6JB4A.png](https://s2.ax1x.com/2019/09/14/n6JB4A.png)

下面开始一步一步进行搭建网络LeNet，由前面介绍的模型构建过程可以知道，其中最为核心的就是搭建模型的网络架构，所以，首先先搭建网络模型，

$$y=wx+b$$

卷积的运算是符合上述公式的，因此，首先构造第一层网络，输入为批量784维的向量，需要首先把它转化为28*28的图像，然后初始化卷积核，进行卷积、激活、池化运算，

```python
X = tf.reshape(X, [-1, 28, 28, 1])
w_1 = tf.get_variable("weights", shape=[5, 5, 1, 6])
b_1 = tf.get_variable("bias", shape=[6])
conv_1 = tf.nn.conv2d(X, w_1, strides=[1, 1, 1, 1], padding="SAME")
act_1 = tf.sigmoid(tf.nn.bias_add(conv_1, b_1))
max_pool_1 = tf.nn.max_pool(act_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
```

然后构建第二层网络，

```python
w_2 = tf.get_variable("weights", shape=[5, 5, 6, 16])
b_2 = tf.get_variable("bias", shape=[16])
conv_2 = tf.nn.conv2d(max_pool_1, w_2, strides=[1, 1, 1, 1], padding="SAME")
act_2 = tf.nn.sigmoid(tf.nn.bias_add(conv_2, b_2))
max_pool_2 = tf.nn.max_pool(act_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
```

到这里，卷积层就搭建完了，下面就开始搭建全连接层。

首先需要把卷积层的输出进行展开成向量，

```python
flatten = tf.reshape(max_pool_2, shape=[-1, 2 * 2 * 16])
```

然后紧接着是3个全连接层，

```python
# 全连接层1
with tf.variable_scope("fc_1") as scope:
    w_fc_1 = tf.get_variable("weight", shape=[2 * 2 * 16, 120])
    b_fc_1 = tf.get_variable("bias", shape=[120], trainable=True)
fc_1 = tf.nn.xw_plus_b(flatten, w_fc_1, b_fc_1)
act_fc_1 = tf.nn.sigmoid(fc_1)

# 全连接层2
with tf.variable_scope("fc_2") as scope:
    w_fc_2 = tf.get_variable("weight", shape=[120, 84])
    b_fc_2 = tf.get_variable("bias", shape=[84], trainable=True)
fc_2 = tf.nn.xw_plus_b(act_fc_1, w_fc_2, b_fc_2)
act_fc_2 = tf.nn.sigmoid(fc_2)

# 全连接层3
with tf.variable_scope("fc_3") as scope:
    w_fc_3 = tf.get_variable("weight", shape=[84, 10])
    b_fc_3 = tf.get_variable("bias", shape=[10], trainable=True)
fc_3 = tf.nn.xw_plus_b(act_fc_2, w_fc_3, b_fc_3)
```

这样就把整个网络模型搭完成了，输入是批量图像X，输出是预测的图像，输出是一个10维向量，每一维的含义是当前数字的概率，选择概率最大的位置，就是图像对应的数字。

完成了网络模型的搭建，它能够将输入图像转化成预测标签进行输出，接下来要做的就是训练和测试部分。

```python
def train():
    # 1. 输入数据的占位符
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, 10])
    
    # 2. 初始化LeNet模型，构造输出标签y_
    le = LeNet()
    y_ = le.create(x)
	
    # 3. 损失函数，使用交叉熵作为损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
	
    # 4. 优化函数，首先声明I个优化函数，然后调用minimize去最小化损失函数
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    
    # 5. summary用于数据保存，用于tensorboard可视化
    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs")
    
    # 6. 构造验证函数，如果对应位置相同则返回true，否则返回false
    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    
    # 7. 通过tf.cast把true、false布尔型的值转化为数值型，分别转化为1和0，然后相加就是判断正确的数量
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 8. 初始化一个saver，用于后面保存训练好的模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 9. 初始化变量
        sess.run((tf.global_variables_initializer()))
        writer.add_graph(sess.graph)
        i = 0
        for epoch in range(5):
            for step in range(1000):
                # 10. feed_dict把数据传递给前面定义的占位符x、y
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                summary, loss_value, _ = sess.run(([merged, loss, train_op]),
                                                  feed_dict={x: batch_xs,
                                                             y: batch_ys})
                print("epoch : {}----loss : {}".format(epoch, loss_value))
                # 11. 记录数据点
                writer.add_summary(summary, i)
                i += 1
                
        # 验证准确率
        test_acc = 0
        test_count = 0
        for _ in range(10):
            batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE)
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            test_acc += acc
            test_count += 1
        print("accuracy : {}".format(test_acc / test_count))
        saver.save(sess, os.path.join("temp", "mode.ckpt"))
```

上述就是训练部分的完整代码，在代码中已经详细的注释了每个部分的功能，分别包含数据记录、损失函数、优化函数、验证函数、训练过程等，然后运行代码可以看到效果，

```shell
...
epoch : 4----loss : 0.07602085173130035
epoch : 4----loss : 0.05565792694687843
epoch : 4----loss : 0.08458487689495087
epoch : 4----loss : 0.012194767594337463
epoch : 4----loss : 0.026294417679309845
epoch : 4----loss : 0.04952147603034973
accuracy : 0.9953125
```

准确率为99.5%，可以看得出，在效果方面，LeNet在某些任务方面并不比深度卷积神经网络差。

打开tensorboard可以直观的看到网络的结构、训练的过程以及训练中数据的变换，

```shell
$ tensorboard --logdir=logs
```

![n6YwrT.gif](https://s2.ax1x.com/2019/09/14/n6YwrT.gif)

通过损失函数的变化过程可以看出，训练过程在2000步左右基本达到了最优解，

![n6J0Nd.png](https://s2.ax1x.com/2019/09/14/n6J0Nd.png)

---

> 更多精彩内容请关注公众号【平凡而诗意】，或者加入我的知识星球【平凡而诗意】~

![n6YcGR.png](https://s2.ax1x.com/2019/09/14/n6YcGR.png)