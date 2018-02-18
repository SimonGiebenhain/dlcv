""" Action recognition - Recurrent Neural Network

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This is pruned version of an original example https://github.com/aymericdamien/TensorFlow-Examples/
for MNIST letter classification

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Adapt this script for purpose of video sequence classification defined in exercise sheet

"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt


def shuffle_in_unision(a,b):
    rng_state = random.getstate()
    random.shuffle(a)
    random.setstate(rng_state)
    random.shuffle(b)


x = pickle.load(open('./X.pickle', 'rb'), encoding='bytes')
labels = np.load('./l.npy')
label_list = [l for l in labels]
shuffle_in_unision(x, label_list)
labels = np.array(label_list)

res = np.zeros((6, 2))
for it in range(6):
    print(res)
    tf.reset_default_graph()

    # Training Parameters
    learning_rate = 0.0001
    training_steps = 1500
    batch_size = 50
    display_step = 250

    # Network Parameters
    num_input = 67*27 # 67 x 27 is size of each frame
    timesteps = 28    # number of timesteps used for classification
    num_hidden = (5*(it + 1))**2  # hidden layer num of features
    num_classes = 10  # 10 actions

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    keep_prob = tf.placeholder(tf.float32)

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }


    def next_batch(x,labels,timesteps,batch_size):
        n = len(x)
        ind = np.random.randint(0,n,batch_size)
        batch_x = [np.reshape(np.transpose(x[i][...,0:timesteps],(2,0,1)),(timesteps,-1)) for i in ind]
        batch_y = np.eye(num_classes)[labels[ind].astype(np.uint8)]
        return np.asarray(batch_x), batch_y


    def next_test_batch(x, labels, timesteps, test_size):
        n = len(x)
        ind = np.random.randint(0, n, test_size)
        batch_x = [np.reshape(np.transpose(x[i][..., 0:timesteps], (2, 0, 1)), (timesteps, -1)) for i in ind]
        batch_y = np.eye(num_classes)[labels[ind].astype(np.uint8)]
        return np.asarray(batch_x), batch_y


    def RNN(x, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.reshape(x, (batch_size * timesteps, 67, 27, 1))

        W_en_1 = tf.get_variable('W_en_1', initializer=tf.random_normal([3, 3, 1, 8], stddev=0.1))
        b_en_1 = tf.get_variable('b_en_1', initializer=tf.constant(0.1, tf.float32, shape=[8]))
        l1 = tf.nn.relu(tf.nn.conv2d(x, W_en_1, [1, 2, 2, 1], padding='SAME') + b_en_1)
        W_en_2 = tf.get_variable('W_en_2', initializer=tf.random_normal([3, 3, 8, 16], stddev=0.1))
        b_en_2 = tf.get_variable('b_en_2', initializer=tf.constant(0.1, tf.float32, shape=[16]))
        l2 = tf.nn.relu(tf.nn.conv2d(l1, W_en_2, [1, 2, 2, 1], padding='SAME') + b_en_2)
        l2 = tf.reshape(l2, (batch_size * timesteps, -1))

        W_fc = tf.get_variable('W_fc', initializer=tf.random_normal([7 * 17 * 16, 7 * 17 * 16], stddev=0.1))
        b_fc = tf.get_variable('b_fc', initializer=tf.constant(0.1, tf.float32, shape=[7 * 17 * 16]))
        l3 = tf.nn.relu(tf.matmul(l2, W_fc) + b_fc)

        feat = tf.reshape(l3, (batch_size, timesteps, -1))
        feat = tf.unstack(feat, timesteps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, feat, dtype=tf.float32)
        drop = [tf.nn.dropout(outputs[i], keep_prob=keep_prob) for i in range(timesteps)]
        # Linear activation, using rnn inner loop last output
        return [tf.matmul(drop[i], weights['out']) + biases['out'] for i in range(timesteps)]


    logits = RNN(X, weights, biases)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.stack([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits[i], labels=Y)) for i in range(timesteps)]))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits[-1], 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    # Start training
    with tf.Session() as sess:
        sum_loss = 0
        sum_acc = 0
        for i in range(5):
            # Run the initializer
            sess.run(init)
            x_train = [x[j] for j in range(75) if j not in range(i*15,(i+1)*15)]
            ind = list(set(range(75)) - set(range(i*15,(i+1)*15)))
            y_train = labels[ind]
            #y_train = [labels[j] for j in range(75) if j not in range(i*15,(i+1)*15)]
            for step in range(1, training_steps+1):
                batch_x, batch_y = next_batch(x_train,y_train,timesteps,batch_size)
                # define the optimization procedure
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y,
                                                                         keep_prob: 1.0})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
            #x_test = np.reshape(np.transpose(x[i*15:(i+1)*15][..., 0:timesteps], (2, 0, 1)), (timesteps, -1))
            #y_test = np.eye(num_classes)[labels[i*15:(i+1)*15].astype(np.uint8)]
            test_x , test_y = next_test_batch(x[i*15:(i+1)*15], labels[i*15:(i+1)*15], timesteps, 50)
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X:test_x, Y:test_y, keep_prob: 1.0})
            print('test loss %s \t test accuracy %s' % (loss, acc))
            sum_loss += loss
            sum_acc += acc
        res[it,0] = sum_loss / 5
        res[it,1] = sum_acc / 5
print(res)