"""
Using tflearn to create 38 convolutional layers + 2 FC layers + Output layer and learn MNIST dataset
First two layers with maxpooling, all convlayers with batch normalization
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)
network = max_pool_2d(network, 2)
# network = local_response_normalization(network)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)
network = max_pool_2d(network, 2)
# network = local_response_normalization(network)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002) #no 20

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = conv_2d(network, 10, 3, activation='relu', regularizer="L2")
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002) #no 38

network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = batch_normalization(network, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002)

network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')