import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


BATCH_SIZE = 64
LR = 0.001
EPOCHS = 1
MAX_STEPS = 100

mnist = input_data.read_data_sets("MNIST", one_hot=True)


class AlexNet(object):
    def __init__(self, num_classes, keep_prob):
        self.num_classes = num_classes
        self.keep_prob = keep_prob

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
        return tf.nn.softmax(fc_3)

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

    def full_connect_layer(self, X, out_filters, name):
        in_filters = X.get_shape()[-1]
        with tf.variable_scope(name) as scope:
            w_fc = tf.get_variable("weight", shape=[in_filters, out_filters])
            b_fc = tf.get_variable("bias", shape=[out_filters], trainable=True)
        fc = tf.nn.xw_plus_b(X, w_fc, b_fc)
        return tf.nn.relu(fc)

    def dropout(self, X, keep_prob):
        return tf.nn.dropout(X, keep_prob)


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


def main(_):
    X = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

    alex_net = AlexNet(10, 0.5)
    y_ = alex_net.create(X)
    train_val(X, y, y_)


if __name__ == '__main__':
    tf.app.run()
