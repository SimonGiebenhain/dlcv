import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
phase_train = tf.placeholder(tf.bool)


def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def bn(x, name):
    return tf.layers.batch_normalization(inputs=x,
                                         beta_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                         gamma_initializer=tf.truncated_normal_initializer(1.0, 0.1),
                                         moving_mean_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                         moving_variance_initializer=tf.truncated_normal_initializer(1.0, 0.1),
                                         training=phase_train,
                                         name=name)


W_conv1 = weight_variable('w_c_1', [5, 5, 1, 32])
b_conv1 = bias_variable('b_c_1', [32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
conv_1 = conv2d(x_image, W_conv1) + b_conv1
norm_1 = bn(conv_1, 'batch_norm_1')
h_conv1 = tf.nn.relu(norm_1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable('w_c_2', [5, 5, 32, 64])
b_conv2 = bias_variable('b_c_2', [64])
conv_2 = conv2d(h_pool1, W_conv2) + b_conv2
norm_2 = bn(conv_2, 'batch_norm_2')
h_conv2 = tf.nn.relu(norm_2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable('w_fc_1', [7 * 7 * 64, 1024])
b_fc1 = bias_variable('b_fc_1', [1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
fc_1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
#norm_fc = bn(fc_1, 'batch_norm_3')
h_fc1 = tf.nn.relu(fc_1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable('w_fc_2', [1024, 10])
b_fc2 = bias_variable('b_fc_2', [10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = np.zeros((11,2))
    for j in range(10):
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, phase_train: True})
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0, phase_train: False})
                print('train accuracy: %s' % train_accuracy)
                res[i,0] = train_accuracy
                test_accuracy = accuracy.eval(
                    feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, phase_train: False})
                print('test accuracy: % s' % test_accuracy)
                res[i,1] = test_accuracy

    test_accuracy = accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, phase_train: False})
    train_accuracy = accuracy.eval(
        feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0, phase_train: False})
    print('train accuracy: %s' % train_accuracy)
    print('train accuracy %s \t test accuracy %s' % (train_accuracy, test_accuracy))
    res[10,0] = train_accuracy
    res[10,1] = test_accuracy
    print(res)
