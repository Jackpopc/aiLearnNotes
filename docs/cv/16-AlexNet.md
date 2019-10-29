---
title: 【动手学计算机视觉】第十六讲：卷积神经网络之AlexNet
---

# 前言

![uv4Na8.png](https://s2.ax1x.com/2019/10/13/uv4Na8.png)

前文详细介绍了卷积神经网络的开山之作<b>LeNet</b>，虽然近几年卷积神经网络非常热门，但是在<b>LeNet</b>出现后的十几年里，在目标识别领域卷积神经网络一直被传统目标识别算法(特征提取+分类器)所压制，直到2012年AlexNet(ImageNet Classification with Deep Convolutional
Neural Networks)在ImageNet挑战赛一举夺魁，使得卷积神经网络再次引起人们的重视，并因此而一发不可收拾，卷积神经网络的研究如雨后春笋一般不断涌现，推陈出新。

<b>AlexNet</b>是以它的第一作者Alex Krizhevsky而命名，这篇文章中也有深度学习领域三位大牛之一的Geoffrey Hinton的身影。AlexNet之所以这么有名气，不仅仅是因为获取比赛冠军这么简单。这么多年，目标识别、目标跟踪相关的比赛层出不穷，获得冠军的团队也变得非常庞大，但是反观一下能够像 AlexNet影响力这么大的，却是寥寥可数。

AlexNet相比于上一代的LeNet它首先在数据集上做了很多工作，

<b>第一点：数据集</b>

我们都知道，限制深度学习的两大因素分别输算力和数据集，AlexNet引入了数据增广技术，对图像进行颜色变换、裁剪、翻转等操作。

<b>第二点：激活函数</b>

在激活函数方面它采用ReLU函数代替Sigmoid函数，前面我用一篇文章详细的介绍了不同激活函数的优缺点，如果看过的同学应该清楚，ReLU激活函数不仅在计算方面比Sigmoid更加简单，而且可以克服Sigmoid函数在接近0和1时难以训练的问题。

<b>第三点：Dropout</b>

这也是AlexNet相对于LeNet比较大一点不同之处，AlexNet引入了Dropout用于解决模型训练过程中容易出现过拟合的问题，此后作者还发表几篇文章详细的介绍Dropout算法，它的引入使得卷积神经网络效果大大提升，直到如今Dropout在模型训练过程中依然被广泛使用。

<b>第四点：模型结构</b>

卷积神经网络的每次迭代，模型架构都会发生非常大的变化，卷积核大小、网络层数、跳跃连接等等，这也是不同卷积神经网络模型之间的区别最明显的一点，由于网络模型比较庞大，一言半语无法描述完整，下面我就来详细介绍一下AlexNet的网络模型。

# AlexNet

[![uv4UIS.png](https://s2.ax1x.com/2019/10/13/uv4UIS.png)](https://imgchr.com/i/uv4UIS)

如果读过前面一片文章应该了解，LeNet是一个5层的卷积神经网络模型，它有两个卷积层和3个全连接层。对比而言，AlexNet是一个8层的卷积升级网络模型，它有5个卷积层和3个全连接层。

我们在搭建一个网络模型的过程中，重点应该关注如下几点：

- 卷积核大小
- 输入输出通道数
- 步长
- 激活函数

关于AlexNet中使用的激活函数前面已经介绍过，它使用的是ReLU激活函数，它5层卷积层除了第一层卷积核为<b>11\*11</b>、第二次为<b>5\*5</b>之外，其余三层均为<b>3\*3</b>，下面就详细介绍一下AlexNet的模型结构，

<b>第一层：卷积层</b>

卷积核大小<b>11\*11</b>，输入通道数根据输入图像而定，输出通道数为<b>96</b>，步长为<b>4</b>。

池化层窗口大小为<b>3\*3</b>，步长为<b>2</b>。

<b>第二层：卷积层</b>

卷积核大小<b>5\*5</b>，输入通道数为<b>96</b>，输出通道数为<b>256</b>，步长为<b>2</b>。

池化层窗口大小为<b>3\*3</b>，步长为<b>2</b>。

<b>第三层：卷积层</b>

卷积核大小<b>3\*3</b>，输入通道数为<b>256</b>，输出通道数为<b>384</b>，步长为<b>1</b>。

<b>第四层：卷积层</b>

卷积核大小<b>3\*3</b>，输入通道数为<b>384</b>，输出通道数为<b>384</b>，步长为<b>1</b>。

<b>第五层：卷积层</b>

卷积核大小<b>3\*3</b>，输入通道数为<b>384</b>，输出通道数为<b>256</b>，步长为<b>1</b>。

池化层窗口大小为<b>3\*3</b>，步长为<b>2</b>。

<b>第六层：全连接层</b>

输入大小为上一层的输出，输出大小为<b>4096</b>。

Dropout概率为<b>0.5</b>。

<b>第七层：全连接层</b>

输入大小为<b>4096</b>，输出大小为<b>4096</b>。

Dropout概率为<b>0.5</b>。

<b>第八层：全连接层</b>

输入大小为<b>4096</b>，输出大小为<b>分类数</b>。

<b>注意：</b>需要注意一点，5个卷积层中前2个卷积层后面都会紧跟一个池化层，而第3、4层卷积层后面没有池化层，而是连续3、4、5层三个卷积层后才加入一个池化层。

# 编程实践

![uv4wGQ.png](https://s2.ax1x.com/2019/10/13/uv4wGQ.png)

在动手实践LeNet文章中，我介绍了网络搭建的过程，这种方式同样适用于除LeNet之外的其他模型的搭建，我们需要首先完成网络模型的搭建，然后再编写训练、验证函数部分。

在前面一篇文章为了让大家更加容易理解tensorflow的使用，更加清晰的看到网络搭建的过程，因此逐行编码进行模型搭建。但是，我们会发现，同类型的网络层之间很多参数是相同的，例如卷积核大小、输出通道数、变量作用于的名称，我们逐行搭建会有很多代码冗余，我们完全可以把这些通用参数作为传入参数提炼出来。因此，本文编程实践中会侧重于代码规范，提高代码的可读性。

编程实践中主要根据tensorflow接口的不同之处把网络架构分为如下4个模块：

- 卷积层
- 池化层
- 全连接层
- Dropout

<b>卷积层</b>

针对卷积层，我们把<b>输入、卷积核大小、输入通道数、步长、变量作用域</b>作为入参，我们使用tensorflow时会发现，我们同样需要知道输入数据的通道数，关于这个变量，我们可以通过获取输入数据的尺寸获得，

```python
def conv_layer(self, X, ksize, out_filters, stride, name):
    in_filters = int(X.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weight = tf.get_variable("weight", [ksize, ksize, in_filters, out_filters])
        bias = tf.get_variable("bias", [out_filters])
    conv = tf.nn.conv2d(X, weight, strides=[1, stride, stride, 1], padding="SAME")
    activation = tf.nn.relu(tf.nn.bias_add(conv, bias))
    return activation
```

上面，我们经过获取权重、偏差，卷积运算，激活函数3个部分完成了卷积模块的实现。AlexNet有5个卷积层，不同层之间的主要区别就体现在<b>conv_layer</b>的入参上面，因此我们只需要修改函数的入参就可以完成不同卷积层的搭建。

<b>池化层</b>

```python
def pool_layer(self, X, ksize, stride):
    return tf.nn.max_pool(X, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME")
```

<b>全连接层</b>

```python
def full_connect_layer(self, X, out_filters, name):
    in_filters = X.get_shape()[-1]
    with tf.variable_scope(name) as scope:
        w_fc = tf.get_variable("weight", shape=[in_filters, out_filters])
        b_fc = tf.get_variable("bias", shape=[out_filters], trainable=True)
    fc = tf.nn.xw_plus_b(X, w_fc, b_fc)
    return tf.nn.relu(fc)
```

<b>Dropout</b>

```python
def dropout(self, X, keep_prob):
    return tf.nn.dropout(X, keep_prob)
```

到这里，我们就完成了卷积层、池化层、全连接层、Dropout四个模块的编写，下面我们只需要把不同模块按照AlexNet的模型累加在一起即可，

<b>模型</b>

```python
def create(self, X):
    X = tf.reshape(X, [-1, 28, 28, 1])
    conv_layer1 = self.conv_layer(X, 11, 96, 4, "Layer1")
    pool_layer1 = self.pool_layer(conv_layer1, 3, 2)

    conv_layer2 = self.conv_layer(pool_layer1, 5, 256, 2, "Layer2")
    pool_layer2 = self.pool_layer(conv_layer2, 3, 2)

    conv_layer3 = self.conv_layer(pool_layer2, 3, 384, 1, "Layer3")
    conv_layer4 = self.conv_layer(conv_layer3, 3, 384, 1, "Layer4")
    conv_layer5 = self.conv_layer(conv_layer4, 3, 256, 1, "Layer5")
    pool_layer = self.pool_layer(conv_layer5, 3, 2)
    _, x, y, z = pool_layer.get_shape()
    full_connect_size = x * y * z
    flatten = tf.reshape(pool_layer, [-1, full_connect_size])
    fc_1 = self.full_connect_layer(flatten, 4096, "fc_1")
    drop1 = self.dropout(fc_1, self.keep_prob)
    fc_2 = self.full_connect_layer(drop1, 4096, "fc_2")
    drop2 = self.dropout(fc_2, self.keep_prob)
    fc_3 = self.full_connect_layer(drop2, self.num_classes, "fc_3")
    return fc_3
```

返回结果是一个<b>1\*m</b>维的向量，其中m是类别数，以本文使用的MNIST为例，输入是一个<b>1\*10</b>的详细，每一个数字对应于索引数字的概率值。

上述就是完整模型的搭建过程，下面我们就需要把输入传入模型，然后获取预测输出，进而构建误差函数进行训练模型。

<b>训练验证</b>

训练验证部分入参有3个，分别是，

- 输入数据
- 标签
- 预测值

其中输入数据和标签为占位符，会在图启动运算时传入真实数据，预测值为模型的输出，

```python
def train_val(X, y, y_):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=LR)
    train_op = optimizer.minimize(loss)
    tf.summary.scalar("loss", loss)

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        i = 0
        for epoch in range(EPOCHS):
            for step in range(MAX_STEPS):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                summary, loss_val, _ = sess.run([merged, loss, train_op],
                                                feed_dict={X: batch_xs, y: batch_ys})
                print("epoch : {}----loss : {}".format(epoch, loss_val))
                writer.add_summary(summary, i)
                i += 1
        saver.save(sess, os.path.join("temp", "mode.ckpt"))

        test_acc = 0
        test_count = 0
        for _ in range(10):
            batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE)
            acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys})
            test_acc += acc
            test_count += 1
        print("accuracy : {}".format(test_acc / test_count))
```

上述就是AlexNet模型搭建和训练过程。

<b>注意：</b>同一个模型在不同的数据集上表现会存在很大差异，例如LeNet是在MNIST的基础上进行搭建和验证的，因此卷积核、步长等这些超参数都已经进行了精心的调节，因此只需要按照模型搭建完成即可得到99%以上的准确率。而AlexNet是在ImageNet的图像上进行调优的，ImageNet的图像相对于MNIST<b>28\*28</b>的图像要大很多，因此卷积核、步长都要大很多，但是这样对于图像较小的MNIST来说就相对较大，很难提取细节特征，因此如果用默认的结构效果甚至比不上20年轻的LeNet。这也是为什么深度学习模型可复制性差的原因，尽管是两个非常类似的任务，同一个模型在两个任务上表现得效果也会存在很大的差异，这需要工程时对其进行反复的调节、优化。

# 完整代码

如果需要完整代码可以在github搜索项目[**aiLearnNotes**](https://github.com/Jackpopc/aiLearnNotes)，或者复制下方链接直接打开，

https://github.com/Jackpopc/aiLearnNotes/blob/master/computer_vision/AlexNet.py

---

> 更多精彩内容，请关注公众号【平凡而诗意】~