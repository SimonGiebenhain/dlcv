import os, cv2, config, h5py, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def load_sample_dataset():
    train_test_file_path = 'train_test_file_list.h5'
    with h5py.File(train_test_file_path, 'r') as hf:
        train_x = np.array(hf.get('train_x')) / 255
        #plt.imshow(train_x[0], cmap='gray')
        #plt.show()
        train_x = np.array(np.split(np.reshape(train_x, (train_x.shape[0], 784)), 463, axis=0))


        y_in = np.squeeze(np.array(hf.get('train_y')))
        train_y = np.zeros((y_in.shape[0], 10))
        train_y[np.arange(y_in.shape[0]), y_in] = 1
        train_y = np.array(np.split(train_y, 463, axis=0))

        test_x = np.array(hf.get('test_x')) / 255
        test_x = np.reshape(test_x, (test_x.shape[0], 784))

        y_test_in = np.squeeze(np.array(hf.get('test_y')))
        test_y = np.zeros((y_test_in.shape[0], 10))
        test_y[np.arange(y_test_in.shape[0]), y_test_in] = 1

    return train_x, train_y, test_x, test_y

train_x,train_y,test_x,test_y = load_sample_dataset()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
  batch_xs = train_x[:, i % 43, :]
  batch_ys = train_y[:, i % 43, :]
  #batch_ys = np.one_hot(train_y[:1000, :], 10)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
print(sess.run(y, feed_dict={x: test_x, y_: test_y}))
print(test_y)