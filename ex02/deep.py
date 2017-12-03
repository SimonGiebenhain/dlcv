import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math as m

a = 20
bb = 1.2

nr_batches = 463
nr_batches_test = 14


def sigmoid_modified(x):
    return 1 / (1 + np.exp(-x * a))


def normalize(I):
    x_queer = np.sum(np.sum(I, axis=2, keepdims=True), axis=1, keepdims=True) / (784 * 255)
    train_x = sigmoid_modified((I / 255) - (x_queer / bb))
    # train_x = I/255 - x_queer
    '''for i in range(10,50):
        plt.subplot(1, 2, 1)
        plt.imshow(I[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(train_x[i], cmap='gray')
        plt.show()'''
    return np.reshape(train_x, (train_x.shape[0], 784))


def load_sample_dataset():
    dataset = 'train_test_file_list.h5'
    with h5py.File(dataset, 'r') as hf:
        train_x_in = np.array(hf.get('train_x'), dtype=np.float64)

        train_y_in = np.squeeze(np.array(hf.get('train_y')))

        test_x_in = np.array(hf.get('test_x'), dtype=np.float64)

        test_y_in = np.squeeze(np.array(hf.get('test_y')))

    return train_x_in, train_y_in, test_x_in, test_y_in


def preprocess_data(x, y, n):
    x_train = np.split(normalize(x), n, axis=0)
    y_train = np.zeros((y.shape[0], 10))
    y_train[np.arange(y.shape[0]), y] = 1
    y_train = np.split(y_train, n, axis=0)
    return x_train, y_train


train_x_in, train_y_in, test_x_in, test_y_in = load_sample_dataset()

train_x, train_y = preprocess_data(train_x_in, train_y_in, nr_batches)
test_x, test_y = preprocess_data(test_x_in, test_y_in, nr_batches_test)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, s):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')

# This method trains the model, as proposed in the google tutorial.
# 2.2.a,b,c,d
# For 2.2.d one has to manually remove the second convolutional layer.
def run_standard_version():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1 / (10 ** 4)).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            train_step.run(feed_dict={x: train_x[i % nr_batches], y_: train_y[i % nr_batches], keep_prob: 0.5})
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: train_x[i % nr_batches], y_: train_y[i % nr_batches], keep_prob: 1.0})
                print('training accuracy %g' % train_accuracy)
            if i % nr_batches == 0:
                rng_state = np.random.get_state()
                np.random.shuffle(train_x_in)
                np.random.set_state(rng_state)
                np.random.shuffle(train_y_in)
                train_x, train_y = preprocess_data(train_x_in, train_y_in, nr_batches)

        test_accuracy = 0.0
        for i in range(len(test_x)):
            test_accuracy += accuracy.eval(feed_dict={x: test_x[i], y_: test_y[i], keep_prob: 1.0})
        test_accuracy = test_accuracy / len(test_x)
        print('test accuracy %s' % test_accuracy)
        train_accuracy = 0.0
        for i in range(len(train_x)):
            train_accuracy += accuracy.eval(feed_dict={x: train_x[i], y_: train_y[i], keep_prob: 1.0})
        train_accuracy = train_accuracy / len(train_x)
        print('train accuracy %s' % train_accuracy)

'''This method evaluates a 3 layer convolutional layer with the following parameters:
k1, k2, k3: kernel sizes
f1,f2,f3: number of features per layer
pool_max: boolean, says wheter max or avg pooling is to be used
s1,s2,s3: stride size of kernels in convolutional layers
xs: training data
ys: testing data
results: place to store the result
n: index for results
This method is to be used from cross_validation().
'''
def hype3(results, k1, k2, f1, f2, pool_max, fc_size, s1, s2, k3, f3, s3, xs, ys, n):
    x_ = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([k1, k1, 1, f1])
    b_conv1 = bias_variable([f1])

    x_image = tf.reshape(x_, [-1, 28, 28, 1])
    size = 28.0

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, s1) + b_conv1)

    if (s1 == 2):
        size = m.ceil(size / 2.0)
    elif(s1 != 1):
        raise ValueError('Stride size not contained in {1,2}')

    if (pool_max):
        h_pool1 = max_pool_2x2(h_conv1)
        size = m.ceil(size / 2.0)
    else:
        h_pool1 = avg_pool_2x2(h_conv1)

    W_conv2 = weight_variable([k2, k2, f1, f2])
    b_conv2 = bias_variable([f2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, s2) + b_conv2)
    if (s2 == 2):
        size = m.ceil(size / 2.0)
    elif(s2 != 1):
        raise ValueError('Stride size not contained in {1,2}')
    '''if (pool_max):
        h_pool2 = max_pool_2x2(h_conv2)
    else:
        h_pool2 = avg_pool_2x2(h_conv2)'''
    h_pool2 = max_pool_2x2(h_conv2)
    size = m.ceil(size / 2.0)

    W_conv3 = weight_variable([k3, k3, f2, f3])
    b_conv3 = bias_variable([f3])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, s3) + b_conv3)

    W_fc1 = weight_variable([size * size * f3, fc_size])
    b_fc1 = bias_variable([fc_size])

    h_pool2_flat = tf.reshape(h_conv3, [-1, size * size * f3])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([fc_size, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc = np.zeros((4, 2))

    with tf.Session() as sess:
        for j in range(len(xs)):
            train_x = np.split(np.concatenate([x for i, x in enumerate(xs) if i != j]), 553, axis=0)
            train_y = np.split(np.concatenate([y for i, y in enumerate(ys) if i != j]), 553, axis=0)
            test_x = np.split(xs[j], 79)
            test_y = np.split(ys[j], 79)

            sess.run(tf.global_variables_initializer())
            for i in range(25000):
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={x_: train_x[i % 553], y_: train_y[i % 553], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x_: train_x[i % 553], y_: train_y[i % 553], keep_prob: 0.5})
                if i % 5000 == 0 & i > 0:
                    test_accuracy = 0
                    for i in range(len(test_x)):
                        test_accuracy += accuracy.eval(feed_dict={x_: test_x[i], y_: test_y[i], keep_prob: 1.0})
                    test_accuracy = test_accuracy / len(test_x)
                    print('test accuracy %s' % test_accuracy)

            train_accuracy = 0.0
            for i in range(len(train_x)):
                train_accuracy += accuracy.eval(feed_dict={x_: train_x[i], y_: train_y[i], keep_prob: 1.0})
            train_accuracy = train_accuracy / len(train_x)
            print('train accuracy %s' % train_accuracy)
            test_accuracy = 0.0
            for i in range(len(test_x)):
                test_accuracy += accuracy.eval(feed_dict={x_: test_x[i], y_: test_y[i], keep_prob: 1.0})
            test_accuracy = test_accuracy / len(test_x)
            print('test accuracy %s' % test_accuracy)
            acc[j, 1] = test_accuracy
            acc[j, 0] = train_accuracy

    tf.reset_default_graph()
    print(np.sum(acc, axis=0)/4)
    results[n] = np.sum(acc, axis=0)/4

'''This method evaluates a 2 layer convolutional layer with the following parameters:
k1, k2: kernel sizes
f1,f2: number of features per layer
pool_max: boolean, says wheter max or avg pooling is to be used
s1,s2: stride size of kernels in convolutional layers
xs: training data
ys: testing data
results: place to store the result
n: index for results
This method is to be used from cross_validation().'''
def hype2(results, k1, k2, f1, f2, pool_max, fc_size, s1, s2, xs, ys, n):
    tf.reset_default_graph()

    x_ = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([k1, k1, 1, f1])
    b_conv1 = bias_variable([f1])

    x_image = tf.reshape(x_, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, s1) + b_conv1)
    if (pool_max):
        h_pool1 = max_pool_2x2(h_conv1)
    else:
        h_pool1 = avg_pool_2x2(h_conv1)

    W_conv2 = weight_variable([k2, k2, f1, f2])
    b_conv2 = bias_variable([f2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, s2) + b_conv2)
    '''if (pool_max):
        h_pool2 = max_pool_2x2(h_conv2)
    else:
        h_pool2 = avg_pool_2x2(h_conv2)'''
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * f2, fc_size])
    b_fc1 = bias_variable([fc_size])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * f2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([fc_size, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc = np.zeros((4, 2))

    with tf.Session() as sess:
        for j in range(len(xs)):
            train_x = np.split(np.concatenate([x for i, x in enumerate(xs) if i != j]), 553, axis=0)
            train_y = np.split(np.concatenate([y for i, y in enumerate(ys) if i != j]), 553, axis=0)
            test_x = np.split(xs[j], 79)
            test_y = np.split(ys[j], 79)

            sess.run(tf.global_variables_initializer())
            for i in range(12500):
                if i % 5000 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={x_: train_x[i % 553], y_: train_y[i % 553], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x_: train_x[i % 553], y_: train_y[i % 553], keep_prob: 0.5})

            train_accuracy = 0.0
            for i in range(len(train_x)):
                train_accuracy += accuracy.eval(feed_dict={x_: train_x[i], y_: train_y[i], keep_prob: 1.0})
            train_accuracy = train_accuracy / len(train_x)
            print('train accuracy %s' % train_accuracy)
            test_accuracy = 0.0
            for i in range(len(test_x)):
                test_accuracy += accuracy.eval(feed_dict={x_: test_x[i], y_: test_y[i], keep_prob: 1.0})
            test_accuracy = test_accuracy / len(test_x)
            print('test accuracy %s' % test_accuracy)
            acc[j, 1] = test_accuracy
            acc[j, 0] = train_accuracy

    tf.reset_default_graph()
    print(np.sum(acc, axis=0)/4)
    results[n] = np.sum(acc, axis=0)/4

'''
This method evaluates a multi layer convolutional layer with the following parameters:
nr_layers: number of layers
size_layers: list, each entry represents the size of convolutional layers stacked to represent that layer
kernel_size: size of kernels
xs: training data
ys: testing data
results: place to store the result
n: index for results
This method is to be used from cross_validation().
'''
def mult_layers(nr_layers, size_layers, kernel_size, xs, ys, results, offset):
    tf.reset_default_graph()
    x_ = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1_head = [weight_variable([kernel_size, kernel_size, 1, 32])]
    W_conv1_tail = [weight_variable([kernel_size, kernel_size, 32, 32]) for i in range(size_layers[0] - 1)]
    W_conv1 = W_conv1_head + W_conv1_tail

    b_conv1 = [bias_variable([32]) for i in range(size_layers[0])]

    x_image = tf.reshape(x_, [-1, 28, 28, 1])

    h_conv1 = []
    for i in range(size_layers[0]):
        if i > 0:
            h_conv1 += [tf.nn.relu(conv2d(h_conv1[i-1], W_conv1[i], 1) + b_conv1[i])]
        else:
            h_conv1 += [tf.nn.relu(conv2d(x_image, W_conv1[i], 1) + b_conv1[i])]


    h_pool1 = max_pool_2x2(h_conv1[size_layers[0] - 1])

    W_conv2_head = [weight_variable([kernel_size, kernel_size, 32, 64])]
    W_conv2_tail = [weight_variable([kernel_size, kernel_size, 64, 64]) for i in range(size_layers[1] - 1)]
    W_conv2 = W_conv2_head + W_conv2_tail

    b_conv2 = [bias_variable([64]) for i in range(size_layers[1])]

    h_conv2 = []
    for i in range(size_layers[1]):
        if i > 0:
            h_conv2 += [tf.nn.relu(conv2d(h_conv2[i - 1], W_conv2[i], 1) + b_conv2[i])]
        else:
            h_conv2 += [tf.nn.relu(conv2d(h_pool1, W_conv2[i], 1) + b_conv2[i])]

    h_pool2 = max_pool_2x2(h_conv2[size_layers[1] - 1])

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc = np.zeros((4, 2))

    with tf.Session() as sess:
        for j in range(len(xs)):
            train_x = np.split(np.concatenate([x for i, x in enumerate(xs) if i != j]), 553, axis=0)
            train_y = np.split(np.concatenate([y for i, y in enumerate(ys) if i != j]), 553, axis=0)
            test_x = np.split(xs[j], 79)
            test_y = np.split(ys[j], 79)

            sess.run(tf.global_variables_initializer())
            for i in range(12500):
                if i % 5000 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={x_: train_x[i % 553], y_: train_y[i % 553], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x_: train_x[i % 553], y_: train_y[i % 553], keep_prob: 0.5})

            train_accuracy = 0.0
            for i in range(len(train_x)):
                train_accuracy += accuracy.eval(feed_dict={x_: train_x[i], y_: train_y[i], keep_prob: 1.0})
            train_accuracy = train_accuracy / len(train_x)
            print('train accuracy %s' % train_accuracy)
            test_accuracy = 0.0
            for i in range(len(test_x)):
                test_accuracy += accuracy.eval(feed_dict={x_: test_x[i], y_: test_y[i], keep_prob: 1.0})
            test_accuracy = test_accuracy / len(test_x)
            print('test accuracy %s' % test_accuracy)
            acc[j, 1] = test_accuracy
            acc[j, 0] = train_accuracy

    tf.reset_default_graph()
    print(np.sum(acc, axis=0) / 4)
    results[offset] = np.sum(acc, axis=0) / 4

'''
This method sets up the environment for a cross validation.
From here all hyper-parameters can be tested. ex. 2.2.f
'''
def cross_validation():
    rng_state = np.random.get_state()
    np.random.shuffle(train_x_in)
    np.random.set_state(rng_state)
    np.random.shuffle(train_y_in)

    xs, ys = preprocess_data(train_x_in[0:19908, :, :], train_y_in[0:19908], 4)

    results = np.zeros((15, 2))

    # TODO: hyper-params: layers, stride, pooling, keepprob, (padding), (relu),
    # TODO: lohnt sich schachteln?

    offset = 0
    for i in range(6, 9):
        k1, k2, k3, f1, f2, f3, max_pool, fc_size, s1, s2, s3 = 5, 5, 0, 2 ** i, 2 ** (i + 1), 0, True, 1024, 1, 1, 0
        print("k1 = %d, k2 = %d, k3 = %d, f1= %d, f2 = %d, f3 = %d, max = %s, fc_size= %d, s1 = %d, s2= %d, s3 = %d" % (
            k1, k2, k3, f1, f2, f3, max_pool, fc_size, s1, s2, s3))
        hype2(results, k1, k2, f1, f2, max_pool, fc_size, s1, s2, xs, ys, offset)
        offset += 1

    print(results)

'''
This method evaluates different step_sizes().
ex. 2.2.e
'''
def step_size():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1 / (10 ** 4)).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        res = np.zeros((7, 21, 2))
        for k in range(len(train_step)):
            sess.run(tf.global_variables_initializer())
            for i in range(15000):
                train_step[k].run(feed_dict={x: train_x[i % nr_batches], y_: train_y[i % nr_batches], keep_prob: 1.0})
                if i % 1000 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: train_x[i % nr_batches], y_: train_y[i % nr_batches], keep_prob: 1.0})
                    test_accuracy = 0.0
                    for j in range(len(test_x)):
                        test_accuracy += accuracy.eval(feed_dict={x: test_x[j], y_: test_y[j], keep_prob: 1.0})
                    test_accuracy = test_accuracy / len(test_x)
                    print('step %d, training accuracy %g, test accuracy %g' % (i, train_accuracy, test_accuracy))
                    res[k, int(i/1000), 0] = train_accuracy
                    res[k, int(i/1000), 1] = test_accuracy
                if i % nr_batches == 0:
                    rng_state = np.random.get_state()
                    np.random.shuffle(train_x_in)
                    np.random.set_state(rng_state)
                    np.random.shuffle(train_y_in)
                    train_x, train_y = preprocess_data(train_x_in, train_y_in, nr_batches)

            test_accuracy = 0.0
            for i in range(len(test_x)):
                test_accuracy += accuracy.eval(feed_dict={x: test_x[i], y_: test_y[i], keep_prob: 1.0})
            test_accuracy = test_accuracy / len(test_x)
            print('test accuracy %s' % test_accuracy)
            train_accuracy = 0.0
            for i in range(len(train_x)):
                train_accuracy += accuracy.eval(feed_dict={x: train_x[i], y_: train_y[i], keep_prob: 1.0})
            train_accuracy = train_accuracy / len(train_x)
            print('train accuracy %s' % train_accuracy)
            res[k, 20, 0] = train_accuracy
            res[k, 20, 1] = test_accuracy
    print(res)

cross_validation()