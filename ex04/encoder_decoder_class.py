import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# normalize all Images, such that each image has zero mean.
# Also invert the colors, such that the background is white and the letters are black.
def normalize_mean(I):
    #I = np.ones(I.shape) - I
    means = np.sum(I, axis=(1,2)) / (I.shape[1] * I.shape[2])
    return I - np.reshape(means, (I.shape[0], 1, 1))


# normalize all images, such that each image has standard deviation of 1
def normalize_stddev(I):
    stddev = np.std(I, axis=(1,2))
    return I / np.reshape(stddev, (I.shape[0], 1, 1))

def shuffle_in_unision(a,b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def make_img_beauty(I):
    I = np.reshape(I, (28,28))
    min = np.min(I)
    max = np.max(I)
    mean = np.mean(I, axis=(0,1))
    #I = I - min / max - min
    I = 10 * (I - 0.1)
    I = 1 / (np.exp(-I) + 1)
    return I


# Load the data. Store inputs in X and labels in Y
X_cyr = np.transpose(np.load("./X_cyr.npy"), (2,0,1))
Y_cyr = np.load("./labels_cyr.npy")

# normalize
X_cyr = normalize_mean(X_cyr)
X_cyr = normalize_stddev(X_cyr)

# Shuffle and split into test and training data (in ratio 5 to 1).
shuffle_in_unision(X_cyr, Y_cyr)
X_cyr = np.reshape(X_cyr, (-1, 28, 28, 1))
X_train = X_cyr[:5000]
X_test = X_cyr[5000:]
Y_train = Y_cyr[:5000]
Y_test = Y_cyr[5000:]


def param_relu(x, a):
    return tf.nn.relu(x) - a * tf.nn.relu(x)


phase_train = tf.placeholder(tf.bool)
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.int32, [None])


# high-level batch normalization
def bn(x, name):
    return tf.layers.batch_normalization(inputs=x,
                                         beta_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                         gamma_initializer=tf.truncated_normal_initializer(1.0, 0.1),
                                         moving_mean_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                         moving_variance_initializer=tf.truncated_normal_initializer(1.0, 0.1),
                                         training=phase_train,
                                         name=name)


alpha = tf.get_variable('alpha', initializer=0.1)

W_en_1 = tf.get_variable('W_en_1', initializer=tf.random_normal([3,3,1,32]), collections=['vars'])
b_en_1 = tf.get_variable('b_en_1', initializer=tf.constant(0.1, tf.float32, shape=[32]))
l_en_1_norm = bn(tf.nn.conv2d(X, W_en_1, [1,2,2,1], padding='SAME') + b_en_1, 'bn_en_1')
l_en_1 = param_relu(l_en_1_norm, alpha)

W_en_2 = tf.get_variable('W_en_2', dtype=tf.float32, initializer=tf.random_normal([3,3,32,64]), collections=['vars'])
b_en_2 = tf.get_variable('b_en_2', initializer=tf.constant(0.1, tf.float32, shape=[64]))
l_en_2_norm = bn(tf.nn.conv2d(l_en_1, W_en_2, [1,2,2,1], padding='SAME') + b_en_2, 'bn_en_2')
l_en_2 = param_relu(l_en_2_norm, alpha)

W_middle_1 = tf.get_variable('W_middle_1', initializer=tf.random_normal([7 * 7 * 64, 512]), collections=['vars'])
b_middle_1 = tf.get_variable('b_middle_1', initializer=tf.random_normal([512]))

flat_1 = tf.reshape(l_en_2, [-1, 7*7*64])
middle_1 = param_relu(tf.matmul(flat_1, W_middle_1) + b_middle_1, alpha)

keep_prob_middle = tf.placeholder(tf.float32)
middle_1_drop = tf.nn.dropout(middle_1, keep_prob_middle)

W_middle_2 = tf.get_variable('W_middle_2', initializer=tf.random_normal([512, 7*7*64]), collections=['vars'])
b_middle_2 = tf.get_variable('b_middle_2', initializer=tf.random_normal([7*7*64]))

middle_2 = tf.reshape(param_relu(tf.matmul(middle_1_drop, W_middle_2) + b_middle_2, alpha), [-1, 7, 7,64])

W_de_1 = tf.get_variable('W_de_1', initializer=tf.random_normal([3,3,32,64]), collections=['vars'])
b_de_1 = tf.get_variable('b_de_1', initializer=tf.constant(0.1, tf.float32, shape=[32]))
batch_size = tf.shape(middle_2)[0]
out_shape_1 = tf.stack([batch_size, 14, 14, 32])
l_de_1_norm = bn(tf.nn.conv2d_transpose(middle_2, W_de_1, out_shape_1, strides=[1,2,2,1], padding='SAME') + b_de_1, 'bn_de_1')
l_de_1 = param_relu(l_de_1_norm, alpha)
W_de_2 = tf.get_variable('W_de_2', initializer=tf.random_normal([3,3,1,32]), collections=['vars'])
b_de_2 = tf.get_variable('b_de_2', initializer=tf.constant(0.1, tf.float32, shape=[1]))
batch_size = tf.shape(l_de_1)[0]
out_shape_2 = tf.stack([batch_size, 28, 28, 1])
l_de_2_norm = bn(tf.nn.conv2d_transpose(l_de_1, W_de_2, out_shape_2, strides=[1,2,2,1], padding='SAME') + b_de_2, 'bn_de_2')
l_de_2 = param_relu(l_de_2_norm, alpha)


def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial, trainable=False)


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial, trainable=False)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


with tf.variable_scope('lat_class'):
    W_conv1 = weight_variable('w_c_1', [5, 5, 1, 32])
    b_conv1 = bias_variable('b_c_1', [32])
    x_image = tf.reshape(l_de_2, [-1, 28, 28, 1])
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
    Y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

lam = 1e-5
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(Y, 15), logits=Y_pred))
weight_decay = cross_entropy + lam * tf.reduce_sum(tf.stack(
    [tf.nn.l2_loss(i) for i in tf.get_collection('vars')])
)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(weight_decay)

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(tf.one_hot(Y, 15), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def train():
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lat_class'))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer(tf.get_collection('vars')))
        saver.restore(sess, "/Users/sigi/PycharmProjects/ex04/weights_2.ckpt")
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection('vars'))
        l = -100
        r = 0
        for i in range(20001):
            l += 100
            r += 100
            if r >= X_train.shape[0]:
                l = 0
                r = 100
                shuffle_in_unision(X_train, Y_train)
            train_step.run(feed_dict={X: X_train[l:r], Y: Y_train[l:r], keep_prob: 1.0, phase_train: True, keep_prob_middle: 0.67})
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(
                        feed_dict={X: X_train[0:200], Y: Y_train[0:200], keep_prob: 1.0, phase_train: False, keep_prob_middle: 1.0})
                test_accuracy = accuracy.eval(feed_dict={X: X_test[0:200], Y: Y_test[0:200], keep_prob: 1.0, phase_train: False, keep_prob_middle: 1.0})
                print('step %d, training accuracy %g, test accuracy %g' % (i, train_accuracy, test_accuracy))
            if i % 5000 == 0 and i > 0:
                saver.save(sess, "/Users/sigi/PycharmProjects/ex04/weights_encoder_decoder_class.ckpt")
        test_accuracy = accuracy.eval(feed_dict={X: X_test, Y: Y_test, keep_prob: 1.0, phase_train: False, keep_prob_middle: 1.0})
        print('test accuracy %s' % test_accuracy)
        train_accuracy = 0.0
        for i in range(int(len(X_train) / 1000)):
            train_accuracy += accuracy.eval(
                feed_dict={X: X_train[i * 1000:(i + 1) * 1000], Y: Y_train[i * 1000:(i + 1) * 1000], keep_prob: 1.0, phase_train: False, keep_prob_middle: 1.0})
        train_accuracy = train_accuracy / len(X_train) * 1000
        print('train accuracy %s' % train_accuracy)
        saver.save(sess, "/Users/sigi/PycharmProjects/ex04/weights_encoder_decoder_class.ckpt")


def display():
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection('vars'))
    # Load the data. Store inputs in X and labels in Y
    X_lat = np.transpose(np.load("./X_lat.npy"), (2, 0, 1))
    Y_lat = np.load("./labels_lat.npy")

    # normalize
    X_lat = normalize_mean(X_lat)
    X_lat = normalize_stddev(X_lat)
    indices_lat = np.ones(14, dtype=np.int32)
    for i in range(14):
        j = 0
        while Y_lat[j] != i:
            j += 1
        indices_lat[i] = j
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.variables_initializer(tf.get_collection('vars')))
        saver.restore(sess, "/Users/sigi/PycharmProjects/ex04/weights_encoder_decoder_class.ckpt")
        res = l_de_2.eval(feed_dict={X:X_test, keep_prob_middle: 1.0, phase_train: False})
        indices_cyr = np.zeros(14, dtype=np.int32)
        #print(accuracy.eval(feed_dict={X:X_test, Y: Y_test, keep_prob:1.0, keep_prob_middle:1.0, phase_train:False}))


        mean_output = np.zeros((14,28,28))
        freq = np.zeros(14)
        for i in range(X_test.shape[0]):
            mean_output[Y_test[i]] += make_img_beauty(np.squeeze(res[i]))
            freq[Y_test[i]] += 1
        mean_output /= np.reshape(freq, (14,1,1))
        for i in range(14):
            j = 0
            while Y_test[j] != i:
                j += 1
            indices_cyr[i] = j
        for i in range(14):
            plt.subplot(3, 14, i+1)
            plt.imshow(np.reshape(X_test[indices_cyr[i]], (28,28)), cmap='gray')
            plt.axis('off')
        for i in range(14):
            plt.subplot(3,14,i+15)
            plt.imshow(np.reshape(X_lat[indices_lat[i]], (28,28)), cmap='gray')
            plt.axis('off')
        for i in range(14):
            plt.subplot(3,14,i+29)
            plt.imshow(np.reshape(res[indices_cyr[i]], (28,28)), cmap='gray')
            plt.axis('off')
        plt.show()


display()