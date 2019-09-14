import os
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST", one_hot=True)
BATCH_SIZE = 64


def visualization(_mnist):
    for i in range(12):
        plt.subplot(3, 4, i+1)
        img = _mnist.train.images[i + 1]
        img = img.reshape(28, 28)
        plt.imshow(img)
    plt.show()


class LeNet(object):
    def __init__(self):
        pass

    def create(self, X):
        X = tf.reshape(X, [-1, 28, 28, 1])
        with tf.variable_scope("layer_1") as scope:
            w_1 = tf.get_variable("weights", shape=[5, 5, 1, 6])
            b_1 = tf.get_variable("bias", shape=[6])
        conv_1 = tf.nn.conv2d(X, w_1, strides=[1, 1, 1, 1], padding="SAME")
        act_1 = tf.sigmoid(tf.nn.bias_add(conv_1, b_1))
        max_pool_1 = tf.nn.max_pool(act_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        with tf.variable_scope("layer_2") as scope:
            w_2 = tf.get_variable("weights", shape=[5, 5, 6, 16])
            b_2 = tf.get_variable("bias", shape=[16])
        conv_2 = tf.nn.conv2d(max_pool_1, w_2, strides=[1, 1, 1, 1], padding="SAME")
        act_2 = tf.sigmoid(tf.nn.bias_add(conv_2, b_2))
        max_pool_2 = tf.nn.max_pool(act_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        flatten = tf.reshape(max_pool_2, shape=[-1, 7 * 7 * 16])
        with tf.variable_scope("fc_1") as scope:
            w_fc_1 = tf.get_variable("weight", shape=[7 * 7 * 16, 120])
            b_fc_1 = tf.get_variable("bias", shape=[120], trainable=True)
        fc_1 = tf.nn.xw_plus_b(flatten, w_fc_1, b_fc_1)
        act_fc_1 = tf.nn.sigmoid(fc_1)

        with tf.variable_scope("fc_2") as scope:
            w_fc_2 = tf.get_variable("weight", shape=[120, 84])
            b_fc_2 = tf.get_variable("bias", shape=[84], trainable=True)
        fc_2 = tf.nn.xw_plus_b(act_fc_1, w_fc_2, b_fc_2)
        act_fc_2 = tf.nn.sigmoid(fc_2)

        with tf.variable_scope("fc_3") as scope:
            w_fc_3 = tf.get_variable("weight", shape=[84, 10])
            b_fc_3 = tf.get_variable("bias", shape=[10], trainable=True)
            tf.summary.histogram("weight", w_fc_3)
            tf.summary.histogram("bias", b_fc_3)
        fc_3 = tf.nn.xw_plus_b(act_fc_2, w_fc_3, b_fc_3)
        return fc_3


def train():
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, 10])
    le = LeNet()
    y_ = le.create(x)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    tf.summary.scalar("loss", loss)
    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer()))
        writer.add_graph(sess.graph)
        i = 0
        for epoch in range(5):
            for step in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                summary, loss_value, _ = sess.run(([merged, loss, train_op]),
                                                  feed_dict={x: batch_xs,
                                                             y: batch_ys})
                print("epoch : {}----loss : {}".format(epoch, loss_value))
                writer.add_summary(summary, i)
                i += 1
        test_acc = 0
        test_count = 0
        for _ in range(10):
            batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE)
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            test_acc += acc
            test_count += 1
        print("accuracy : {}".format(test_acc / test_count))
        saver.save(sess, os.path.join("temp", "mode.ckpt"))


train()
