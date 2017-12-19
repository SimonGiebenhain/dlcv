import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def leakyRELU(x, a):
    return tf.nn.relu(x) - a * tf.nn.relu(-x)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

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


init = tf.constant(0, dtype=tf.float32)
a = tf.get_variable('alpha', initializer=init, trainable=False)

W_conv1 = weight_variable('w_c_1', [5, 5, 1, 32])
b_conv1 = bias_variable('b_c_1', [32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = leakyRELU(conv2d(x_image, W_conv1) + b_conv1, a)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable('w_c_2',[5, 5, 32, 64])
b_conv2 = bias_variable('b_c_2', [64])

h_conv2 = leakyRELU(conv2d(h_pool1, W_conv2) + b_conv2, a)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable('w_fc_1', [7 * 7 * 64, 1024])
b_fc1 = bias_variable('b_fc_1',[1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = leakyRELU(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, a)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable('w_fc_2', [1024, 10])
b_fc2 = bias_variable('b_fc_2', [10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    res = np.zeros((11,2))
    for j in range(11):
        sess.run(tf.global_variables_initializer())
        sess.run(a.assign(j * 0.1))
        print('Alpha is %s' % a.eval())
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('train accuracy: %s' % train_accuracy)

        x_test = np.split(mnist.test.images, 10)
        y_test = np.split(mnist.test.labels, 10)
        test_accuracy = 0
        for i in range(10):
            test_accuracy += accuracy.eval(feed_dict={x: x_test[i], y_: y_test[i], keep_prob: 1.0})
        test_accuracy /= 10

        x_train = np.split(mnist.train.images, 55)
        y_train = np.split(mnist.train.labels, 55)
        train_accuracy = 0
        for i in range(55):
            train_accuracy += accuracy.eval(
               feed_dict={x: x_train[i], y_: y_train[i], keep_prob: 1.0})
        train_accuracy /= 55
        print('train accuracy %s \t test accuracy %s' % (train_accuracy, test_accuracy))
        res[j,0] = train_accuracy
        res[j,1] = test_accuracy
    print(res)