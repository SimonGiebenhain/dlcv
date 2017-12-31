import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
phase_train = tf.placeholder(tf.bool)

#encapsulates get_variable with xavier initialization
def weight_variable(name, shape):
    #weights of fully connected layer
    if (len(shape) == 2):
        fan_in = tf.constant(shape[0], dtype=tf.float32, shape=())
        #fan_out = tf.constant(shape[1], dtype=tf.float32, shape=())
        #initial = tf.random_normal(shape) / tf.sqrt(fan_in)
    #wieghts of convolutional layer
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

# high-level batch normalization
def bn(x, name):
    return tf.layers.batch_normalization(inputs=x,
                                         beta_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                         gamma_initializer=tf.truncated_normal_initializer(1.0, 0.1),
                                         moving_mean_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                         moving_variance_initializer=tf.truncated_normal_initializer(1.0, 0.1),
                                         training=phase_train,
                                         name=name)

# encapsulates a residual block consisting of two conv-layers (conv + bn + relu)
# in case, that the number of features are increased with this residual layer,
# an additional 1x1-conv-layer is introduced to make up for the increased features.
def res_block(x, feats_in, feats_out, nr, feature_increase):
    sub1 = res_sub_block(x, feats_in, feats_out, nr, 1)
    sub2 = res_sub_block(sub1, feats_out, feats_out, nr, 2)
    if feature_increase:
        w = weight_variable('feature_increase_%d' % nr, [1, 1, feats_in, feats_out])
        return conv2d(x, w) + sub2
    else:
        return x + sub2

# encapsulates one of the conv layers of the residual layer
def res_sub_block(x, feats_in, feats_out, nr, sub_nr):
    w1 = weight_variable('w%d_block%d' % (sub_nr, nr), [3, 3, feats_in, feats_out])
    b1 = bias_variable('b%d_block%d' % (sub_nr, nr), [feats_out])
    conv1 = conv2d(x, w1) + b1
    conv1_norm = bn(conv1, 'bn%d_block%d' % (sub_nr, nr))
    return tf.nn.relu(conv1_norm)


curr_in = tf.reshape(x, [-1, 28, 28, 1])
# 7 residual layers with 32 features, each consisting of 2 convolutional layers
for i in range(7):
    if i == 0:
        curr_in = res_block(curr_in, 1, 32, i+1, True)
    else:
        curr_in = res_block(curr_in, 32, 32, i+1, False)
curr_in = max_pool_2x2(curr_in)

# 7 residual layers with 48 features, each consisting of 2 convolutional layers
for i in range(7):
    if i == 0:
        curr_in = res_block(curr_in, 32, 48, i+8, True)
    else:
        curr_in = res_block(curr_in, 48, 48, i+8, False)
curr_in = max_pool_2x2(curr_in)

# 7 residual layers with 64 features, each consisting of 2 convolutional layers
for i in range(7):
    if i == 0:
        curr_in = res_block(curr_in,48, 64, i+15, True)
    else:
        curr_in = res_block(curr_in, 64, 64, i+15, False)


# In the following the features extracted in the 21 residual layers (42 conv-layers),
# are combined with a fully connected layer of size 512.
# Dropout is applied on the way of the fully connected layer to the 10 class nodes
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

# Here in addition the the cross entropy, weight decay with lambda=1e-5 is applied.
# In the weight decay alle weight and bias variables are included.
lam = 1e-5
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
weight_decay = cross_entropy + lam * tf.reduce_sum(tf.stack(
    [tf.nn.l2_loss(i) for i in tf.get_collection('vars')])
)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(weight_decay)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.variables_initializer(tf.get_collection('vars')))

    for j in range(20000):
        # Probably better performance with bigger batch size (because bn works better).
        # However this slows down training siginficantly, and my computer sucks.
        batch = mnist.train.next_batch(150)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, phase_train: True})
        if j % 100 == 0:
            print(j)
        # Evalute training, validation and test sets after every 1000 training cycles.
        if j % 1000 == 0:
            x_test = np.split(mnist.test.images, 10)
            y_test = np.split(mnist.test.labels, 10)
            test_accuracy = 0
            for i in range(10):
                test_accuracy += accuracy.eval(
                    feed_dict={x: x_test[i], y_: y_test[i], keep_prob: 1.0, phase_train: False})
            test_accuracy /= 10
            x_vali = np.split(mnist.validation.images, 5)
            y_vali = np.split(mnist.validation.labels, 5)
            vali_accuracy = 0
            for i in range(5):
                vali_accuracy += accuracy.eval(
                    feed_dict={x: x_vali[i], y_: y_vali[i], keep_prob: 1.0, phase_train: False})
            vali_accuracy /= 5
            x_train = np.split(mnist.train.images, 55)
            y_train = np.split(mnist.train.labels, 55)
            train_accuracy = 0
            for i in range(55):
                train_accuracy += accuracy.eval(
                    feed_dict={x: x_train[i], y_: y_train[i], keep_prob: 1.0, phase_train: False})
            train_accuracy /= 55
            print('step %d: train accuracy %s \t validation accuracy %s \t test accuracy %s' % (j,
                                                                        train_accuracy, vali_accuracy, test_accuracy))

    x_test = np.split(mnist.test.images, 10)
    y_test = np.split(mnist.test.labels, 10)
    test_accuracy = 0
    for i in range(10):
        test_accuracy += accuracy.eval(feed_dict={x: x_test[i], y_: y_test[i], keep_prob: 1.0, phase_train: False})
    test_accuracy /= 10
    x_vali = np.split(mnist.validation.images, 5)
    y_vali = np.split(mnist.validation.labels, 5)
    vali_accuracy = 0
    for i in range(5):
        vali_accuracy += accuracy.eval(feed_dict={x: x_vali[i], y_: y_vali[i], keep_prob: 1.0, phase_train: False})
    vali_accuracy /= 5
    x_train = np.split(mnist.train.images, 55)
    y_train = np.split(mnist.train.labels, 55)
    train_accuracy = 0
    for i in range(55):
        train_accuracy += accuracy.eval(
            feed_dict={x: x_train[i], y_: y_train[i], keep_prob: 1.0, phase_train: False})
    train_accuracy /= 55
    print('train accuracy %s \t validation accuracy %s \t test accuracy %s' % (train_accuracy, vali_accuracy, test_accuracy))
