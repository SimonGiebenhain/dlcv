import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# normalize all Images, such that each image has zero mean.
# Also invert the colors, such that the background is white and the letters are black.
def normalize_mean(I):
    #I = np.ones(I.shape) - I
    means = np.sum(I, axis=(1, 2)) / (I.shape[1] * I.shape[2])
    return I - np.reshape(means, (I.shape[0], 1, 1))


# normalize all images, such that each image has standard deviation of 1
def normalize_stddev(I):
    stddev = np.std(I, axis=(1, 2))
    return I / np.reshape(stddev, (I.shape[0], 1, 1))


def shuffle_in_unision(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.uint8, shape=[None])


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


with tf.variable_scope('lat_class'):
    W_conv1 = weight_variable('w_c_1', [5, 5, 1, 32])
    b_conv1 = bias_variable('b_c_1', [32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable('w_c_2', [5, 5, 32, 64])
    b_conv2 = bias_variable('b_c_2', [64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable('w_fc_1', [7 * 7 * 64, 1024])
    b_fc1 = bias_variable('b_fc_1', [1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable('w_fc_2', [1024, 15])
    b_fc2 = bias_variable('b_fc_2', [15])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y_, 15), logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf.one_hot(y_, 15), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()  # Load the data. Store inputs in X and labels in Y.
# Reorder data, such that the first dimension contains the different images.
X_lat = np.transpose(np.load("./X_lat.npy"), (2, 0, 1))
Y_lat = np.load("./labels_lat.npy")

# normalize
X_lat = normalize_mean(X_lat)
X_lat = normalize_stddev(X_lat)

# Shuffle and split into test and training data (in ratio 5 to 1).
shuffle_in_unision(X_lat, Y_lat)
X_lat = np.reshape(X_lat, (-1, 28, 28, 1))
X_train = np.concatenate((X_lat[:5000], np.random.standard_normal(size=(5000, 28, 28, 1))), axis=0)
X_test = X_lat[5000:]
Y_train = np.concatenate((Y_lat[:5000], 15 * np.ones(shape=(5000))), axis=0)
Y_test = Y_lat[5000:]

def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l = -100
        r = 0
        for i in range(20000):
            l += 100
            r += 100
            if r >= X_train.shape[0]:
                l = 0
                r = 100
                shuffle_in_unision(X_train, Y_train)
            train_step.run(feed_dict={x: X_train[l:r], y_: Y_train[l:r], keep_prob: 0.5})
            if i % 50 == 0:
                train_accuracy = 0.0
                for j in range(int(len(X_train) / 1000)):
                    train_accuracy += accuracy.eval(
                        feed_dict={x: X_train[j * 1000:(j + 1) * 1000], y_: Y_train[j * 1000:(j + 1) * 1000],
                                   keep_prob: 1.0})
                train_accuracy = train_accuracy / len(X_train) * 1000
                test_accuracy = 0.0
                for j in range(int(len(X_test) / 35)):
                    test_accuracy += accuracy.eval(
                        feed_dict={x: X_test[j * 35:(j + 1) * 35], y_: Y_test[j * 35:(j + 1) * 35], keep_prob: 1.0})
                test_accuracy = test_accuracy / len(X_test) * 35
                print('step %d, training accuracy %g, test accuracy %g' % (i, train_accuracy, test_accuracy))

                save_path = saver.save(sess, "/Users/sigi/PycharmProjects/ex04/weights_2.ckpt")
                print("Model saved in file: %s" % save_path)

        test_accuracy = 0.0
        for j in range(int(len(X_test) / 35)):
            test_accuracy += accuracy.eval(
                feed_dict={x: X_test[j * 35:(j + 1) * 35], y_: Y_test[j * 35:(j + 1) * 35], keep_prob: 1.0})
        test_accuracy = test_accuracy / len(X_test) * 35
        print('test accuracy %s' % test_accuracy)
        train_accuracy = 0.0
        for i in range(int(len(X_train) / 1000)):
            train_accuracy += accuracy.eval(
                feed_dict={x: X_train[i * 1000:(i + 1) * 1000], y_: Y_train[i * 1000:(i + 1) * 1000], keep_prob: 1.0})
        train_accuracy = train_accuracy / len(X_train) * 1000
        print('train accuracy %s' % train_accuracy)

        # save_path = saver.save(sess, "/Users/sigi/PycharmProjects/ex04/weights.ckpt")
        # print("Model saved in file: %s" % save_path)


def classify(X):
    with tf.Session() as sess:
        saver.restore(sess, "/Users/sigi/PycharmProjects/ex04/weights_2.ckpt")
        return y_conv.eval(feed_dict={x: np.array(X), keep_prob: 1.0})

def test():
    with tf.Session() as sess:
        saver.restore(sess, "/Users/sigi/PycharmProjects/ex04/weights_2.ckpt")
        test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
        print('test accuracy %s' % test_accuracy)
        train_accuracy = 0.0
        for i in range(int(len(X_train) / 1000)):
            train_accuracy += accuracy.eval(
                feed_dict={x: X_train[i * 1000:(i + 1) * 1000], y_: Y_train[i * 1000:(i + 1) * 1000], keep_prob: 1.0})
        train_accuracy = train_accuracy / len(X_train) * 1000
        print('train accuracy %s' % train_accuracy)


train()