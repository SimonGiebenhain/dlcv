import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# normalize all Images, such that each image has zero mean.
# Also invert the colors, such that the background is white and the letters are black.
def normalize_mean(I):
    # I = np.ones(I.shape, dtype=np.float32) - I
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


# Load the data. Store inputs in X and labels in Y
X_cyr = np.transpose(np.load("./X_cyr.npy"), (2, 0, 1))
Y_cyr = np.load("./labels_cyr.npy")
X_lat = np.transpose(np.load("./X_lat.npy"), (2, 0, 1))
Y_lat = np.load("./labels_lat.npy")

# normalize
X_cyr = normalize_mean(X_cyr)
X_cyr = normalize_stddev(X_cyr)
X_lat = normalize_mean(X_lat)
X_lat = normalize_stddev(X_lat)


def compute_clusters(nr):
    return np.array([pics for (i, pics) in enumerate(X_lat) if Y_lat[i] == nr])


indices_lat = np.zeros(14, dtype=np.int16)
for i in range(14):
    j = 0
    while Y_lat[j] != i:
        j += 1
    indices_lat[i] = j
examples = np.array([X_lat[indices_lat[i]] for i in range(14)])

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

Y = tf.placeholder(tf.float32, shape=[None, 28,28])


# high-level batch normalization
def bn(x, name):
    return tf.layers.batch_normalization(inputs=x,
                                         beta_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                         gamma_initializer=tf.truncated_normal_initializer(1.0, 0.1),
                                         moving_mean_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                         moving_variance_initializer=tf.truncated_normal_initializer(1.0, 0.1),
                                         training=phase_train,
                                         name=name)


alpha = tf.get_variable('alpha', initializer=0.1, collections=['vars'])

W_middle_2 = tf.get_variable('W_middle_2', initializer=tf.random_normal([512, 7 * 7 * 64]), collections=['vars'],
                             trainable=False)
b_middle_2 = tf.get_variable('b_middle_2', initializer=tf.random_normal([7 * 7 * 64]), collections=['vars'], trainable=False)


W_de_1 = tf.get_variable('W_de_1', initializer=tf.random_normal([3, 3, 32, 64]), collections=['vars'], trainable=False)
b_de_1 = tf.get_variable('b_de_1', initializer=tf.constant(0.1, tf.float32, shape=[32]), collections=['vars'], trainable=False)

W_de_2 = tf.get_variable('W_de_2', initializer=tf.random_normal([3, 3, 1, 32]), collections=['vars'], trainable=False)
b_de_2 = tf.get_variable('b_de_2', initializer=tf.constant(0.1, tf.float32, shape=[1]), collections=['vars'], trainable=False)


W_fc = tf.get_variable('W_fc', initializer=tf.random_normal([28 * 28, 28 * 28]), collections=['vars'], trainable=False)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


latent = tf.get_variable('latent', initializer=tf.random_normal([1, 512]))
sess.run(latent.initializer)

bn_middle_2 = bn(tf.matmul(latent, W_middle_2) + b_middle_2, 'bn_middle_2')
middle_2 = tf.reshape(param_relu(bn_middle_2, alpha), [1, 7, 7, 64])

out_shape_1 = tf.stack([1, 14, 14, 32])
l_de_1_norm = bn(tf.nn.conv2d_transpose(middle_2, W_de_1, out_shape_1, strides=[1, 2, 2, 1], padding='SAME') + b_de_1,
                 'bn_de_1')
l_de_1 = param_relu(l_de_1_norm, alpha)

out_shape_2 = tf.stack([1, 28, 28, 1])
l_de_2_norm = bn(tf.nn.conv2d_transpose(l_de_1, W_de_2, out_shape_2, strides=[1, 2, 2, 1], padding='SAME') + b_de_2,
                 'bn_de_2')
l_de_2 = param_relu(l_de_2_norm, alpha)

lat_pred = tf.matmul(tf.reshape(l_de_2, (1, 28 * 28)), W_fc)
lat_pred = tf.reshape(lat_pred, (1, 28, 28))
loss = tf.reduce_mean(tf.square(Y - tf.reshape(lat_pred, (1, 28, 28))))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

for j in range(14):
    plt.subplot(2, 14, j+1)
    plt.imshow(examples[j], cmap='gray')
for j in range(14):
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection('vars'))
    saver.restore(sess, "/Users/sigi/PycharmProjects/ex04/weights_encoder_decoder_random.ckpt")

    for i in range(100):
        train_step.run(feed_dict={Y: compute_clusters(j), phase_train: True})
        if i % 1000 == 0:
            loss_ = loss.eval(feed_dict={Y: compute_clusters(j), phase_train: False})
            print('nr %d, step %d, loss %g' % (j,i, loss_))
    plt.subplot(2, 14, 15+j)
    plt.imshow(np.squeeze(lat_pred.eval(feed_dict={Y: compute_clusters(j), phase_train: False})), cmap='gray')
plt.show()