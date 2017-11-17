import os, cv2, config, h5py, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

a = 20
bb = 1.2

def sigmoid_modified(x):
    return 1 / (1 + np.exp(-x * a))

def nomrmalize(I):
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
        train_x = np.array(np.split(nomrmalize(np.array(hf.get('train_x'), dtype=np.float64)), 463, axis=0), dtype=np.float64)

        y_in = np.squeeze(np.array(hf.get('train_y')))
        train_y = np.zeros((y_in.shape[0], 10))
        train_y[np.arange(y_in.shape[0]), y_in] = 1
        train_y = np.array(np.split(train_y, 463, axis=0))

        test_x = nomrmalize(np.array(hf.get('test_x'), dtype=np.float64))

        y_test_in = np.squeeze(np.array(hf.get('test_y')))
        test_y = np.zeros((y_test_in.shape[0], 10))
        test_y[np.arange(y_test_in.shape[0]), y_test_in] = 1

    return train_x, train_y, test_x, test_y


train_x,train_y,test_x,test_y = load_sample_dataset()




x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

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

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  j = 0
  for i in range(20000):
    batch= [train_x[j, :, :], train_y[j, :, :]]
    j = (j + 1) % 463
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: test_x, y_: test_y, keep_prob: 1.0}))