# 前言

激活函数不仅对于卷积神经网络非常重要，在传统机器学习中也具备着举足轻重的地位，是卷积神经网络模型中必不可少的一个单元，要理解激活函数，需要从2个方面进行讨论：

- 什么是激活函数？
- 为什么需要激活函数？

**什么是激活函数？**

对于神经网络，一层的输入通过加权求和之后输入到一个函数，被这个函数作用之后它的非线性性增强，这个作用的函数即是激活函数。

**为什么需要激活函数？**

试想一下，对于神经网络而言，如果没有激活函数，每一层对输入进行加权求和后输入到下一层，直到从第一层输入到最后一层一直采用的就是线性组合的方式，根据线性代数的知识可以得知，第一层的输入和最后一层的输出也是呈线性关系的，换句话说，这样的话无论中加了多少层都没有任何价值，这是第一点。

第二点是由于如果没有激活函数，输入和输出是呈线性关系的，但是现实中很多模型都是非线性的，通过引入激活函数可以增加模型的非线性，使得它更好的拟合非线性空间。

目前激活函数有很多，例如阶跃函数、逻辑函数、双曲正切函数、ReLU函数、Leaky ReLU函数、高斯函数、softmax函数等，虽然函数有很多，但是比较常用的主要就是逻辑函数和ReLU函数，在大多数卷积神经网络模型中都是采用这两种，当然也有部分会采用Leaky ReLU函数和双曲正切函数，本文就介绍一下这4个激活函数长什么样？有什么优缺点？在tensorflow中怎么使用？

## Sigmoid

Sigmoid函数的方程式为：

$$f(x)=\sigma(x)=\frac{1}{1+e^{-x}}$$

![maP6SO.png](https://s2.ax1x.com/2019/08/21/maP6SO.png)

绘图程序：

```python
def sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = 1 / (1+np.exp(-x))
    plt.plot(x, y)
    plt.grid()
    plt.show()
```

Sigmoid函数就是前面所讲的逻辑函数，它的主要优点如下：

- 能够将函数压缩至区间[0, 1]之间，保证数据稳定，波动幅度小
- 容易求导

缺点：

- 函数在两端的饱和区梯度趋近于0，当反向传播时容易出现梯度消失或梯度爆炸
- 输出不是0均值(zero-centered)，这样会导致，如果输入为正，那么导数总为正，反向传播总往正方向更新，如果输入为负，那么导数总为负，反向传播总往负方向更新，收敛速度缓慢
- 对于幂运算和规模较大的网络运算量较大

# 双曲正切函数

双曲正切函数方程式：

$$f(x)=\tanh (x)=\frac{\left(e^{x}-e^{-x}\right)}{\left(e^{x}+e^{-x}\right)}$$

![maPrY6.png](https://s2.ax1x.com/2019/08/21/maPrY6.png)

绘图程序：

```python
def tanh():
    x = np.arange(-10, 10, 0.1)
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    plt.plot(x, y)
    plt.grid()
    plt.show()
```

可以看出，从图形上看双曲正切和Sigmoid函数非常类似，但是从纵坐标可以看出，Sigmoid被压缩在[0, 1]之间，而双曲正切函数在[-1, 1]之间，两者的不同之处在于，Sigmoid是非0均值(zero-centered)，而双曲是0均值的，它的相对于Sigmoid的优点就很明显了：

- 提高了训练效率

虽然双曲正切函数解决了Sigmoid函数非0均值的问题，但是它依然没有解决Sigmoid的两位两个问题，这也是tanh的缺点：

- 梯度消失和梯度爆炸
- 对于幂运算和规模较大的网络运算量较大

# ReLU

ReLU函数方程式：

$$f(x)=\left\{\begin{array}{ll}{0} & {\text { for } x<0} \\ {x} & {\text { for } x \geq 0}\end{array}\right.$$

![maPDFx.png](https://s2.ax1x.com/2019/08/21/maPDFx.png)

绘图程序：

```python
def relu():
    x = np.arange(-10, 10, 0.1)
    y = np.where(x<0, 0, x)
    plt.plot(x, y)
    plt.grid()
    plt.show()
```

线性整流函数(Rectified Linear Unit，ReLU)，对比于Sigmoid函数和双曲正切函数的优点如下：

- 梯度不饱和，收敛速度快
- 减轻反向传播时梯度弥散的问题
- 由于不需要进行指数运算，因此运算速度快、复杂度低

虽然解决了Sigmoid和双曲正切函数的缺点，但是它也有明显的不足：

- 输出不是0均值(zero-centered)
- 对参数初始化和学习率非常敏感，设置不当容易造成神经元坏死现象，也就是有些神经元永远不会被激活(由于负部梯度永远为0造成)

# Leaky ReLU

Leaky ReLU函数方程式：

$$f(x)=\left\{\begin{array}{ll}{0.01 x} & {\text { for } x<0} \\ {x} & {\text { for } x \geq 0}\end{array}\right.$$

![maPsfK.png](https://s2.ax1x.com/2019/08/21/maPsfK.png)

绘图程序：

```python
def leaky_relu():
    x = np.arange(-2, 2, 0.1)
    y = np.where(x<0, 0.01*x, x)
    plt.plot(x, y)
    plt.grid()
    plt.show()
```

为了解决ReLU函数神经元坏死现象，Leaky ReLU函数在输入为负是引入了一个(0, 1)之间的常数，使得输入为负时梯度不为0。虽然Leaky ReLU解决了ReLU的这个严重问题，但是它并不总是比ReLU函数效果好，在很多情况下ReLU函数的效果还是更胜一筹。

# tensorflow激活函数使用

tensorflow中激活函数在tf.nn模块下，例如，

```python
tf.nn.relu
tf.nn.sigmoid
tf.nn.tanh
tf.nn.leaky_relu
```

其中relu、sigmoid、tanh函数的参数完全相同，leaky_relu多一个输入参数，就是斜率，默认值为0.2，以relu函数为例介绍一下tensorflow中激活函数的使用，

```python
features = tf.nn.max_poo()
tf.nn.relu(features, name=None)
```

tensorflow中激活函数输入有两个参数：

- features：输入的特征张量，也就是前一层池化层或者卷积层输出的结果，数据类型限制在float32, float64, int32, uint8, int16, int8, int64, float16, uint16, uint32, uint64
- name：运算的名称，这个可以自行命名

**完整代码**链接：[aiLearnNotes](https://github.com/Jackpopc/aiLearnNotes/blob/master/computer_vision/activation.py)