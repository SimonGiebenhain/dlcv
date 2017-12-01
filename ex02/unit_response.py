import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
'''print(sess.run(W_conv1))
print(sess.run(b_conv1))
print(sess.run(W_conv2))
print(sess.run(b_conv1))
print(sess.run(W_fc1))
print(sess.run(b_fc1))
print(sess.run(W_fc2))
print(sess.run(b_fc2))'''

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

unit = tf.reshape(h_conv1, (784,32))
unit = tf.norm(unit, ord=2, axis=0) ** 2
lam = 2
step_size = 1e-2
loss = [lam * (tf.norm(X, ord=2) ** 2) - unit[i] for i in range(32)]
train_step = [tf.train.GradientDescentOptimizer(step_size).minimize(loss[i]) for i in range(32)]

res = np.zeros((32,28,28))
for i in range(32):
    sess.run(X.initializer)

    for j in range(5):
        gain = 1
        last_loss = sess.run(loss[i], {keep_prob: 1})
        while gain > 0:
            for _ in range(50):
                train_step[i].run({keep_prob: 1})
                curr_loss = sess.run(loss[i], {keep_prob: 1})
                gain = last_loss - curr_loss
                last_loss = curr_loss
                #plt.imshow(np.reshape(sess.run(X, {keep_prob: 1}), (28,28)), cmap='gray')
                #plt.show()
            curr_loss = sess.run(loss[i], {keep_prob: 1})
            #print(np.reshape(sess.run(X, {keep_prob: 1}), (28, 28)))
            print(curr_loss)
        train_step[i] = tf.train.GradientDescentOptimizer(step_size/(10**(1+j))).minimize(loss[i])
    res[i, :, :] = np.reshape(sess.run(X, {keep_prob: 1}), (28,28))

for i in range(32):
    plt.subplot(8,4,i + 1)
    plt.imshow(res[i, :, :], cmap='gray')
    plt.axis('off')
plt.show()
