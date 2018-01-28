import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt


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


def generateTrainingset(sizeMultiplier):
    # Load dataset
    X_cyr = np.load('data/X_cyr.npy') #shape = (28, 28, 5798)
    labels_cyr = np.load('data/labels_cyr.npy') #shape = (5798,)
    X_lat = np.load('data/X_lat.npy') #shape = (28, 28, 6015)
    labels_lat = np.load('data/labels_lat.npy')

    # REORDER DIMENSIONS OF DATA
    X_cyr = np.transpose(X_cyr, (2,0,1))
    X_lat = np.transpose(X_lat, (2,0,1))

    # STANDARDIZE DATA
    for i in range(0, len(X_cyr)-1):
        X_cyr[i] = (X_cyr[i] - np.mean(X_cyr[i])) / np.std(X_cyr[i])
    for i in range(0, len(X_lat)-1):
        X_lat[i] = (X_lat[i] - np.mean(X_lat[i])) / np.std(X_lat[i])

    # MATCH TRAINING DATA (AND EXTEND TRAINING SET)
    nr_classes = len(np.unique(labels_cyr)) # nr. of classes is equal in both label arrays
    stepsCyr = np.insert(np.where(labels_cyr[:-1] != labels_cyr[1:])[0], 0, 0)
    stepsLat = np.insert(np.where(labels_lat[:-1] != labels_lat[1:])[0], 0, 0)
    amountCyr = np.insert(np.diff(stepsCyr), 0, stepsCyr[1])
    amountLat = np.insert(np.diff(stepsLat), 0, stepsLat[1])
    sum = 0
    for i in range(0, nr_classes-1):
        sum += min(amountCyr[i], amountLat[i])
    cyrData = np.empty((sum*sizeMultiplier, 28, 28, 1)) # create dataset with 4x the training data size
    latData = np.empty((sum*sizeMultiplier, 28, 28, 1))
    curr_pos = 0
    for _ in range(sizeMultiplier):
        for i in range(0, nr_classes-1):
            # check which side has less images
            if amountCyr[i] < amountLat[i]:
                permuted_indexes = np.random.permutation(amountLat[i])
                for j in range(0, amountCyr[i]-1):
                    cyrData[curr_pos,:,:,0] = X_cyr[stepsCyr[i]+j]
                    latData[curr_pos,:,:,0] = X_lat[permuted_indexes[j]+stepsLat[i]]
                    curr_pos += 1
            else:
                permuted_indexes = np.random.permutation(amountCyr[i])
                for j in range(0, amountLat[i]-1):
                    cyrData[curr_pos,:,:,0] = X_cyr[permuted_indexes[j]+stepsCyr[i]]
                    latData[curr_pos,:,:,0] = X_lat[stepsLat[i]+j]
                    curr_pos += 1

    return cyrData, latData


A_cyr, B_lat = generateTrainingset(10)
# Shuffle and split into test and training data (in ratio 5 to 1).
shuffle_in_unision(A_cyr, B_lat)
A_train = A_cyr[:50000]
A_test = A_cyr[50000:]
B_train = B_lat[:50000]
B_test = B_lat[50000:]


def param_relu(x, a):
    return tf.nn.relu(x) - a * tf.nn.relu(x)

phase_train = tf.placeholder(tf.bool)
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 28, 28, 1])


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

W_en_1 = tf.get_variable('W_en_1', initializer=tf.random_normal([3,3,1,32]), collections=['vars'])
b_en_1 = tf.get_variable('b_en_1', initializer=tf.constant(0.1, tf.float32, shape=[32]), collections=['vars'])
l_en_1_norm = bn(tf.nn.conv2d(X, W_en_1, [1,2,2,1], padding='SAME') + b_en_1, 'bn_en_1')
l_en_1 = param_relu(l_en_1_norm, alpha)

W_en_2 = tf.get_variable('W_en_2', dtype=tf.float32, initializer=tf.random_normal([3,3,32,64]), collections=['vars'])
b_en_2 = tf.get_variable('b_en_2', initializer=tf.constant(0.1, tf.float32, shape=[64]), collections=['vars'])
l_en_2_norm = bn(tf.nn.conv2d(l_en_1, W_en_2, [1,2,2,1], padding='SAME') + b_en_2, 'bn_en_2')
l_en_2 = param_relu(l_en_2_norm, alpha)

W_middle_1 = tf.get_variable('W_middle_1', initializer=tf.random_normal([7 * 7 * 64, 512]), collections=['vars'])
b_middle_1 = tf.get_variable('b_middle_1', initializer=tf.random_normal([512]), collections=['vars'])

flat_1 = tf.reshape(l_en_2, [-1, 7*7*64])
middle_1 = param_relu(bn(tf.matmul(flat_1, W_middle_1) + b_middle_1, 'bn_middle_1'), alpha)

keep_prob_middle = tf.placeholder(tf.float32)
middle_1_drop = tf.nn.dropout(middle_1, keep_prob_middle)

W_middle_2 = tf.get_variable('W_middle_2', initializer=tf.random_normal([512, 7*7*64]), collections=['vars'])
b_middle_2 = tf.get_variable('b_middle_2', initializer=tf.random_normal([7*7*64]), collections=['vars'])

middle_2 = tf.reshape(param_relu(bn(tf.matmul(middle_1_drop, W_middle_2) + b_middle_2, 'bn_middle_2'), alpha), [-1, 7, 7,64])

W_de_1 = tf.get_variable('W_de_1', initializer=tf.random_normal([3,3,32,64]), collections=['vars'])
b_de_1 = tf.get_variable('b_de_1', initializer=tf.constant(0.1, tf.float32, shape=[32]), collections=['vars'])
batch_size = tf.shape(middle_2)[0]
out_shape_1 = tf.stack([batch_size, 14, 14, 32])
l_de_1_norm = bn(tf.nn.conv2d_transpose(middle_2, W_de_1, out_shape_1, strides=[1,2,2,1], padding='SAME') + b_de_1, 'bn_de_1')
l_de_1 = param_relu(l_de_1_norm, alpha)
W_de_2 = tf.get_variable('W_de_2', initializer=tf.random_normal([3,3,1,32]), collections=['vars'])
b_de_2 = tf.get_variable('b_de_2', initializer=tf.constant(0.1, tf.float32, shape=[1]), collections=['vars'])
batch_size = tf.shape(l_de_1)[0]
out_shape_2 = tf.stack([batch_size, 28, 28, 1])
l_de_2_norm = bn(tf.nn.conv2d_transpose(l_de_1, W_de_2, out_shape_2, strides=[1,2,2,1], padding='SAME') + b_de_2, 'bn_de_2')
l_de_2 = param_relu(l_de_2_norm, alpha)

W_fc = tf.get_variable('W_fc', initializer=tf.random_normal([28*28,28*28]), collections=['vars'])
b_fc = tf.get_variable('b_fc', initializer=tf.random_normal([1, 28*28]), collections=['vars'])
lat_pred = tf.matmul(tf.reshape(l_de_2, (-1, 28*28)), W_fc) + b_fc
lat_pred = tf.reshape(lat_pred, (-1, 28, 28, 1))
#lat_pred = tf.matmul(tf.reshape(l_de_2, (-1, 28,28)), tf.tile(tf.reshape(W_fc, (1,28,28)), [tf.shape(l_de_2)[0],1,1]))

loss = tf.reduce_mean(tf.square(Y - lat_pred))
#losses = ([tf.square(lat_actual[i] - tf.reshape(lat_pred, (1,28,28))) for i in range(batch_size)])
#loss = sum([tf.reduce_mean(losses[i]) for i in range(batch_size)]) / len(losses)

#lam = 1e-5
#weight_decay = loss + lam * tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('vars')]))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)


def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer(tf.get_collection('vars')))
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection('vars'))
        l = -100
        r = 0
        for i in range(20000):
            l += 100
            r += 100
            if r >= A_train.shape[0]:
                l = 0
                r = 100
            train_step.run(feed_dict={X: A_train[l:r], Y: B_train[l:r], phase_train: True, keep_prob_middle: 0.67})
            if i % 500 == 0:
                train_accuracy = loss.eval(
                        feed_dict={X: A_train[0:200], Y: B_train[0:200], phase_train: False, keep_prob_middle: 1.0})
                test_accuracy = loss.eval(feed_dict={X: A_test[0:200], Y: B_test[0:200], phase_train: False, keep_prob_middle: 1.0})
                print('step %d, training accuracy %g, test accuracy %g' % (i, train_accuracy, test_accuracy))
            if i % 2000 == 0 and i > 0:
                saver.save(sess, "saves/weights_encoder_decoder_random.ckpt")
        test_accuracy = loss.eval(feed_dict={X: A_test, Y: B_test, phase_train: False, keep_prob_middle: 1.0})
        print('test accuracy %s' % test_accuracy)
        train_accuracy = 0.0
        for i in range(int(len(A_train) / 1000)):
            train_accuracy += loss.eval(
                feed_dict={X: A_train[i * 1000:(i + 1) * 1000], Y: B_train[i * 1000:(i + 1) * 1000], phase_train: False, keep_prob_middle: 1.0})
        train_accuracy = train_accuracy / len(A_train) * 1000
        print('train accuracy %s' % train_accuracy)
        saver.save(sess, "saves//weights_encoder_decoder_random.ckpt")


def train_for_latent_space():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer(tf.get_collection('vars')))
        saver = tf.train.Saver(tf.get_collection('vars'))
        l = -100
        r = 0
        for i in range(20000):
            l += 100
            r += 100
            if r >= A_train.shape[0]:
                l = 0
                r = 100
            train_step.run(feed_dict={X: A_train[l:r], Y: B_train[l:r], phase_train: True, keep_prob_middle: 0.67})
            if i % 500 == 0:
                train_accuracy = loss.eval(
                        feed_dict={X: A_train[0:200], Y: B_train[0:200], phase_train: False, keep_prob_middle: 1.0})
                test_accuracy = loss.eval(feed_dict={X: A_test[0:200], Y: B_test[0:200], phase_train: False, keep_prob_middle: 1.0})
                print('step %d, training accuracy %g, test accuracy %g' % (i, train_accuracy, test_accuracy))
            if i % 2000 == 0 and i > 0:
                saver.save(sess, "saves/weights_encoder_decoder_latent.ckpt")
        test_accuracy = loss.eval(feed_dict={X: A_test, Y: B_test, phase_train: False, keep_prob_middle: 1.0})
        print('test accuracy %s' % test_accuracy)
        train_accuracy = 0.0
        for i in range(int(len(A_train) / 1000)):
            train_accuracy += loss.eval(
                feed_dict={X: A_train[i * 1000:(i + 1) * 1000], Y: B_train[i * 1000:(i + 1) * 1000], phase_train: False, keep_prob_middle: 1.0})
        train_accuracy = train_accuracy / len(A_train) * 1000
        print('train accuracy %s' % train_accuracy)
        saver.save(sess, "saves/weights_encoder_decoder_latent.ckpt")


def display():
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection('vars'))
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.variables_initializer(tf.get_collection('vars')))
        saver.restore(sess, "saves/weights_encoder_decoder_random.ckpt")
        res = lat_pred.eval(feed_dict={X:A_test, keep_prob_middle: 1.0, phase_train: False})
        for a in range(14):
            plt.subplot(4, 14, a+1)
            plt.imshow(np.reshape(A_test[a], (28,28)), cmap='gray')
            plt.axis('off')
        for b in range(14):
            plt.subplot(4,14,b+15)
            plt.imshow(np.reshape(B_test[b], (28,28)), cmap='gray')
            plt.axis('off')
        for c in range(14):
            plt.subplot(4,14,c+29)
            plt.imshow(np.reshape(res[c], (28,28)), cmap='gray')
            plt.axis('off')

        plt.savefig('test.png')
        plt.show()

display()








