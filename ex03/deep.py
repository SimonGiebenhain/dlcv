import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
phase_train = tf.placeholder(tf.bool)


def weight_variable(name, shape):
    if (len(shape) == 2):
        fan_in = tf.constant(shape[0], dtype=tf.float32, shape=())
        #fan_out = tf.constant(shape[1], dtype=tf.float32, shape=())
        #initial = tf.random_normal(shape) / tf.sqrt(fan_in)
    else:
        fan_in = tf.constant(shape[0]*shape[1]*shape[2], dtype=tf.float32, shape=())
        #fan_out = tf.constant(shape[0]*shape[1]*shape[3], dtype=tf.float32, shape=())
        #initial = tf.random_normal((shape[0]*shape[1]*shape[2], shape[0]*shape[1]*shape[3])) / tf.sqrt(fan_in)
    initial = tf.random_normal(shape) / tf.sqrt(fan_in / 2)

    return tf.get_variable(name, initializer=initial, collections=['vars'])


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


def res_block(x, feats_in, feats_out, nr, feature_increase):
    sub1 = res_sub_block(x, feats_in, feats_out, nr, 1)
    sub2 = res_sub_block(sub1, feats_out, feats_out, nr, 2)
    if feature_increase:
        w = weight_variable('feature_increase_%d' % nr, [1, 1, feats_in, feats_out])
        return conv2d(x, w) + sub2
    else:
        return x + sub2


def res_sub_block(x, feats_in, feats_out, nr, sub_nr):
    w1 = weight_variable('w%d_block%d' % (sub_nr, nr), [3, 3, feats_in, feats_out])
    b1 = bias_variable('b%d_block%d' % (sub_nr, nr), [feats_out])
    conv1 = conv2d(x, w1) + b1
    conv1_norm = bn(conv1, 'bn%d_block%d' % (sub_nr, nr))
    return tf.nn.relu(conv1_norm)


curr_in = tf.reshape(x, [-1, 28, 28, 1])
for i in range(3):
    if i == 0:
        curr_in = res_block(curr_in, 1, 16, i+1, True)
    else:
        curr_in = res_block(curr_in, 16, 16, i+1, False)
curr_in = max_pool_2x2(curr_in)

for i in range(3):
    if i == 0:
        curr_in = res_block(curr_in, 16, 32, i+4, True)
    else:
        curr_in = res_block(curr_in, 32, 32, i+4, False)
curr_in = max_pool_2x2(curr_in)

for i in range(3):
    if i == 0:
        curr_in = res_block(curr_in, 32, 64, i+7, True)
    else:
        curr_in = res_block(curr_in, 64, 64, i+7, False)


W_fc1 = weight_variable('w_fc_1', [7 * 7 * 64, 512])
b_fc1 = bias_variable('b_fc_1', [512])

flat = tf.reshape(curr_in, [-1, 7 * 7 * 64])
fc_1 = tf.matmul(flat, W_fc1) + b_fc1
norm_fc = bn(fc_1, 'batch_norm_3')
h_fc1 = tf.nn.relu(norm_fc)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable('w_fc_2', [512, 10])
b_fc2 = bias_variable('b_fc_2', [10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

lam = 1e-5
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
weight_decay = cross_entropy + lam * tf.reduce_sum(tf.stack(
    [tf.nn.l2_loss(i) for i in tf.get_collection('vars')])
)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(weight_decay)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.variables_initializer(tf.get_collection('vars')))

    for i in range(5000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1, phase_train: True})
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0, phase_train: False})
            print('train accuracy: %s' % train_accuracy)

    test_accuracy = accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, phase_train: False})
    train_accuracy = accuracy.eval(
        feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0, phase_train: False})
    print('train accuracy: %s' % train_accuracy)
    print('train accuracy %s \t test accuracy %s' % (train_accuracy, test_accuracy))
