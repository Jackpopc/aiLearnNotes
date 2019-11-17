import numpy as np
import tensorflow as tf


class GoogLeNet(object):
    def __init__(self, num_classes, keep_prob):
        self.num_classes = num_classes
        self.keep_prob = keep_prob

    @staticmethod
    def inception_block(X, c1, c2, c3, c4, name):
        in_channels = int(X.get_shape()[-1])
        # 线路1
        with tf.variable_scope('conv1X1_{}'.format(name)) as scope:
            weight = tf.get_variable("weight", [1, 1, in_channels, c1])
            bias = tf.get_variable("bias", [c1])
        p1_1 = tf.nn.conv2d(X, weight, strides=[1, 1, 1, 1], padding="SAME")
        p1_1 = tf.nn.relu(tf.nn.bias_add(p1_1, bias))

        # 线路2
        with tf.variable_scope('conv2X1_{}'.format(name)) as scope:
            weight = tf.get_variable("weight", [1, 1, in_channels, c2[0]])
            bias = tf.get_variable("bias", [c2[0]])
        p2_1 = tf.nn.conv2d(X, weight, strides=[1, 1, 1, 1], padding="SAME")
        p2_1 = tf.nn.relu(tf.nn.bias_add(p2_1, bias))
        p2_shape = int(p2_1.get_shape()[-1])
        with tf.variable_scope('conv2X2_{}'.format(name)) as scope:
            weight = tf.get_variable("weight", [3, 3, p2_shape, c2[1]])
            bias = tf.get_variable("bias", [c2[1]])
        p2_2 = tf.nn.conv2d(p2_1, weight, strides=[1, 1, 1, 1], padding="SAME")
        p2_2 = tf.nn.relu(tf.nn.bias_add(p2_2, bias))

        # 线路3
        with tf.variable_scope('conv3X1_{}'.format(name)) as scope:
            weight = tf.get_variable("weight", [1, 1, in_channels, c3[0]])
            bias = tf.get_variable("bias", [c3[0]])
        p3_1 = tf.nn.conv2d(X, weight, strides=[1, 1, 1, 1], padding="SAME")
        p3_1 = tf.nn.relu(tf.nn.bias_add(p3_1, bias))
        p3_shape = int(p3_1.get_shape()[-1])
        with tf.variable_scope('conv3X2_{}'.format(name)) as scope:
            weight = tf.get_variable("weight", [5, 5, p3_shape, c3[1]])
            bias = tf.get_variable("bias", [c3[1]])
        p3_2 = tf.nn.conv2d(p3_1, weight, strides=[1, 1, 1, 1], padding="SAME")
        p3_2 = tf.nn.relu(tf.nn.bias_add(p3_2, bias))

        # 线路4
        p4_1 = tf.nn.max_pool(X, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")
        p4_shape = int(p4_1.get_shape()[-1])
        with tf.variable_scope('conv4X2_{}'.format(name)) as scope:
            weight = tf.get_variable("weight", [1, 1, p4_shape, c4])
            bias = tf.get_variable("bias", [c4])
        p4_2 = tf.nn.conv2d(p4_1, weight, strides=[1, 1, 1, 1], padding="SAME")
        p4_2 = tf.nn.relu(tf.nn.bias_add(p4_2, bias))
        return tf.concat([p1_1, p2_2, p3_2, p4_2], axis=3)

    def conv_layer(self, X, ksize, out_filters, stride, name):
        in_filters = int(X.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weight = tf.get_variable("weight", [ksize, ksize, in_filters, out_filters])
            bias = tf.get_variable("bias", [out_filters])
        conv = tf.nn.conv2d(X, weight, strides=[1, stride, stride, 1], padding="SAME")
        activation = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return activation

    def pool_layer(self, X, ksize, stride):
        return tf.nn.max_pool(X, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME")

    def linear(self, X, out_filters, name):
        in_filters = X.get_shape()[-1]
        with tf.variable_scope(name) as scope:
            w_fc = tf.get_variable("weight", shape=[in_filters, out_filters])
            b_fc = tf.get_variable("bias", shape=[out_filters], trainable=True)
        fc = tf.nn.xw_plus_b(X, w_fc, b_fc)
        return tf.nn.relu(fc)

    def create(self, X):
        # 模块1
        module1_1 = self.conv_layer(X, 7, 64, 2, "module1_1")
        pool_layer1 = self.pool_layer(module1_1, 3, 2)

        # 模块2
        module2_1 = self.conv_layer(pool_layer1, 1, 64, 1, "modul2_1")
        module2_2 = self.conv_layer(module2_1, 3, 192, 1, "module2_2")
        pool_layer2 = self.pool_layer(module2_2, 3, 2)

        # 模块3
        module3a = self.inception_block(pool_layer2, 64, (96, 128), (16, 32), 32, "3a")
        module3b = self.inception_block(module3a, 128, (128, 192), (32, 96), 64, "3b")
        pool_layer3 = self.pool_layer(module3b, 3, 2)

        # 模块4
        module4a = self.inception_block(pool_layer3, 192, (96, 208), (16, 48), 64, "4a")
        module4b = self.inception_block(module4a, 160, (112, 224), (24, 64), 64, "4b")
        module4c = self.inception_block(module4b, 128, (128, 256), (24, 64), 64, "4c")
        module4d = self.inception_block(module4c, 112, (144, 288), (32, 64), 64, "4d")
        module4e = self.inception_block(module4d, 256, (160, 320), (32, 128), 128, "4e")
        pool_layer4 = self.pool_layer(module4e, 3, 2)

        # 模块5
        module5a = self.inception_block(pool_layer4, 256, (160, 320), (32, 128), 128, "5a")
        module5b = self.inception_block(module5a, 384, (192, 384), (48, 128), 128, "5b")

        pool_layer5 = tf.nn.avg_pool(module5b, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID")
        flatten = tf.reshape(pool_layer5, [-1, 1024])
        dropout = tf.nn.dropout(flatten, keep_prob=self.keep_prob)
        linear = self.linear(dropout, self.num_classes, 'linear')
        return tf.nn.softmax(linear)


def main():
    X = np.random.normal(size=(5, 224, 224, 3))
    images = tf.placeholder("float", [5, 224, 224, 3])
    googlenet = GoogLeNet(1000, 0.4)
    writer = tf.summary.FileWriter("logs")
    with tf.Session() as sess:
        model = googlenet.create(images)
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        prob = sess.run(model, feed_dict={images: X})
        print(sess.run(tf.argmax(prob, 1)))


if __name__ == '__main__':
    main()