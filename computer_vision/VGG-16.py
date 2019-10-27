import numpy as np
import tensorflow as tf


class VGG(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

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

    def max_pool(self, X):
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def full_connect_layer(self, X, out_filters, name):
        in_filters = X.get_shape()[-1]
        with tf.variable_scope(name) as scope:
            w_fc = tf.get_variable("weight", shape=[in_filters, out_filters])
            b_fc = tf.get_variable("bias", shape=[out_filters], trainable=True)
        fc = tf.nn.xw_plus_b(X, w_fc, b_fc)
        return tf.nn.relu(fc)

    def create(self, X):
        conv_block1 = self.conv_block(X, 2, 1, 64)
        max_pool1 = self.max_pool(conv_block1)

        conv_block2 = self.conv_block(max_pool1, 2, 2, 128)
        max_pool2 = self.max_pool(conv_block2)

        conv_block3 = self.conv_block(max_pool2, 3, 3, 256)
        max_pool3 = self.max_pool(conv_block3)

        conv_block4 = self.conv_block(max_pool3, 3, 4, 512)
        max_pool4 = self.max_pool(conv_block4)

        conv_block5 = self.conv_block(max_pool4, 3, 5, 512)
        max_pool5 = self.max_pool(conv_block5)

        _, x, y, z = max_pool5.get_shape()
        full_connect_size = x * y * z
        flatten = tf.reshape(max_pool5, [-1, full_connect_size])
        fc_1 = self.full_connect_layer(flatten, 4096, "fc6")
        print(fc_1.get_shape())
        fc_2 = self.full_connect_layer(fc_1, 4096, "fc7")
        print(fc_2.get_shape())
        fc_3 = self.full_connect_layer(fc_2, self.num_classes, "fc8")
        print(fc_3.get_shape())
        return tf.nn.softmax(fc_3)


def load_pre_train_weight():
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


if __name__ == '__main__':
    main()