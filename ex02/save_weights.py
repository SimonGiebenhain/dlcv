import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

a = 20
bb = 1.2

nr_batches = 463
nr_batches_test = 14

def sigmoid_modified(x):
    return 1 / (1 + np.exp(-x * a))

def normalize(I):
    x_queer = np.sum(np.sum(I, axis=2, keepdims=True), axis=1, keepdims=True) / (784 * 255)
    train_x = sigmoid_modified((I / 255) - (x_queer/bb))
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

W_conv1 = weight_variable('w_c_1', [5, 5, 1, 32])
b_conv1 = bias_variable('b_c_1', [32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable('w_c_2',[5, 5, 32, 64])
b_conv2 = bias_variable('b_c_2', [64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable('w_fc_1', [7 * 7 * 64, 1024])
b_fc1 = bias_variable('b_fc_1',[1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

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
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        train_step.run(feed_dict={x: train_x[i % nr_batches], y_: train_y[i % nr_batches], keep_prob: 0.5})
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: train_x[i % nr_batches], y_: train_y[i % nr_batches], keep_prob: 1.0})
            test_accuracy = 0.0
            for j in range(len(test_x)):
                test_accuracy += accuracy.eval(feed_dict={x: test_x[j], y_: test_y[j], keep_prob: 1.0})
            test_accuracy = test_accuracy / len(test_x)
            print('step %d, training accuracy %g, test accuracy %g' % (i, train_accuracy, test_accuracy))
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

    print(sess.run(W_conv1))
    print(sess.run(b_conv1))
    print(sess.run(W_conv2))
    print(sess.run(b_conv1))
    print(sess.run(W_fc1))
    print(sess.run(b_fc1))
    save_path = saver.save(sess, "/Users/sigi/PycharmProjects/ex02/weights.ckpt")
    print("Model saved in file: %s" % save_path)

