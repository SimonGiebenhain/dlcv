import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

a = 20

def sigmoid_modified(x):
    return 1 / (1 + np.exp(-x * a))

def weight_variable(name, shape, trainable):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial, trainable=trainable)

def bias_variable(name, shape, trainable):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial, trainable=trainable)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable('w_c_1', [5, 5, 1, 32], False)
b_conv1 = bias_variable('b_c_1', [32], False)

W_conv2 = weight_variable('w_c_2',[5, 5, 32, 64], False)
b_conv2 = bias_variable('b_c_2', [64], False)


W_fc1 = weight_variable('w_fc_1', [7 * 7 * 64, 1024], False)
b_fc1 = bias_variable('b_fc_1',[1024], False)


W_fc2 = weight_variable('w_fc_2', [1024, 10], False)
b_fc2 = bias_variable('b_fc_2', [10], False)



# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
saver.restore(sess, "/Users/sigi/PycharmProjects/ex02/weights.ckpt")
print("Model restored.")

X = tf.get_variable('x', initializer=(tf.truncated_normal((784,), mean=0.5, stddev=0.2)))
sess.run(X.initializer)

x_image = tf.reshape(X, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.reshape(tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2), (10,))

# This medthod computes the response of a single unit
def unit_response(unit):
    lam = 5
    step_size = 1
    loss = lam * abs(1 - (tf.norm(X, ord=2) ** 2)) - unit
    train_step = tf.train.GradientDescentOptimizer(step_size).minimize(loss)
    sess.run(X.initializer)

    for j in range(5):
        gain = 1
        last_loss = sess.run(loss, {keep_prob: 1})
        it = 0
        while gain > 0:
            it += 1
            if it > 1e3:
                print('Maximum iterations reached')
                break
            curr_loss = sess.run(loss, {keep_prob: 1})
            if abs(curr_loss) > 10e13:
                print('Divergeing values!')
                break
            for _ in range(1):
                train_step.run({keep_prob: 1})
                curr_loss = sess.run(loss, {keep_prob: 1})
                gain = last_loss - curr_loss
                last_loss = curr_loss
                # plt.imshow(np.reshape(sess.run(X, {keep_prob: 1}), (28,28)), cmap='gray')
                # plt.show()
        train_step = tf.train.GradientDescentOptimizer(step_size / (10 ** (1 + j))).minimize(loss)
        curr_loss = sess.run(loss, {keep_prob: 1})
        print(curr_loss)
    return np.reshape(sess.run(X, {keep_prob: 1}), (28, 28))

#With this method the response of convolution layers can be computed.
#However parameters of unit_response() might have to be tweaked.
# Especially in the case of node = h_pool2 or h_conv2, nans have to be removed.
def unit_response_inner(node):
    print(node.shape)

    res = np.zeros((node.shape[3],28,28))
    for i in range(node.shape[3]):
        if i == 17:
            continue
        print('Feature: %d' % i)
        unit = tf.reshape(node, (-1, node.shape[3]))[:, i]
        unit = tf.norm(unit, ord=2, axis=0) ** 2
        res[i, :, :] = unit_response(unit)
    print(node.shape)
    for i in range(node.shape[3]):
        if np.max(np.abs(res[i, :, :])) > 1e10:
            continue
        h = int(node.shape[3]) / 8
        plt.subplot(8,h,i + 1)
        plt.imshow(res[i, :, :], cmap='gray')
        plt.axis('off')
    plt.show()

#This method computes the unit response for the final layer.
def unit_response_final_layer():
    res = np.zeros((10,28,28))
    unit = tf.reshape(y_conv, (10,))
    # unit = tf.norm(unit, ord=2, axis=0) ** 2
    lam = 0.1
    step_size = 1
    for i in range(10):
        print('Class: %d' % i)
        sess.run(X.initializer)
        loss = lam * abs(1 - (tf.norm(X, ord=2) ** 2)) - unit[i]
        train_step = tf.train.GradientDescentOptimizer(step_size).minimize(loss)

        for j in range(3):
            gain = 1
            last_loss = sess.run(loss, {keep_prob: 1})
            while gain > 0:
                for _ in range(1):
                    train_step.run({keep_prob: 1})
                    curr_loss = sess.run(loss, {keep_prob: 1})
                    gain = last_loss - curr_loss
                    last_loss = curr_loss
                    # plt.imshow(np.reshape(sess.run(X, {keep_prob: 1}), (28,28)), cmap='gray')
                    # plt.show()
                curr_loss = sess.run(loss, {keep_prob: 1})
            train_step = tf.train.GradientDescentOptimizer(step_size / (10 ** (1 + j))).minimize(loss)
        img = sess.run(X, {keep_prob: 1})
        res[i, :, :] = np.reshape(sigmoid_modified(img - np.mean(img)), (28, 28))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(res[i, :, :], cmap='gray')
        plt.axis('off')
    plt.show()

unit_response_inner(h_conv1)
unit_response_inner(h_pool1)
unit_response_final_layer()
