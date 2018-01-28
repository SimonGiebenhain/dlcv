import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt


# Load dataset
X_cyr = np.load('data/X_cyr.npy') #shape = (28, 28, 5798)
labels_cyr = np.load('data/labels_cyr.npy') #shape = (5798,)
X_lat = np.load('data/X_lat.npy') #shape = (28, 28, 6015)
labels_lat = np.load('data/labels_lat.npy')

# DISPLAY A SYMBOL
# from PIL import Image
# img = Image.fromarray(np.uint8(X_cyr[:,:,0] * 255), 'L')
# img.show()

# THE CLASS DISTRIBUTION IS NOT THE SAME IN CYR AND LAT
# for i in range(1,5000, 100):
#     print(labels_cyr[i], "  ", labels_lat[i])

# REORDER DIMENSIONS OF DATA
X_cyr = np.transpose(X_cyr, (2,0,1))
X_lat = np.transpose(X_lat, (2,0,1))

# STANDARDIZE DATA
for i in range(0, len(X_cyr)-1):
    X_cyr[i] = (X_cyr[i] - np.mean(X_cyr[i])) / np.std(X_cyr[i])
for i in range(0, len(X_lat)-1):
    X_lat[i] = (X_lat[i] - np.mean(X_lat[i])) / np.std(X_lat[i])

# MATCH TRAINING DATA (AND EXTEND TRAINING SET)
sizeMultiplier = 8
nr_classes = len(np.unique(labels_cyr)) # nr. of classes should be equal in both label arrays
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


# NETWORK ARCHITECTURE

batch_size = 50
input_images = tf.placeholder(tf.float32, [None, 28, 28, 1]) #np.float32 ?
output_images = tf.placeholder(tf.float32, [None, 28, 28, 1]) #np.float32 ?
conv1 = tf.layers.conv2d(inputs=input_images,
                         filters=8,
                         kernel_size=(3, 3),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1,
                         filters=8,
                         kernel_size=(3, 3),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
# conv_output = tf.contrib.layers.flatten(conv2)
# dense_layer = tf.layers.dense(inputs=conv_output,
#                               units=28,
#                               activation=tf.nn.relu)
# dense_output = tf.layers.dense(input=dense_layer,
#                                units=())

tconv1 = tf.layers.conv2d_transpose(inputs=conv2,
                                    filters=8,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu)
tconv2 = tf.layers.conv2d_transpose(inputs=tconv1,
                                    filters=1,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu)

loss = tf.nn.l2_loss(output_images - tconv2)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.0005,
                                           global_step=global_step,
                                           decay_steps=int(20000/(2*batch_size)),
                                           decay_rate=0.95,
                                           staircase=True)
trainer = tf.train.RMSPropOptimizer(learning_rate)
training_step = trainer.minimize(loss)
# training_step = tf.train.AdamOptimizer(0.01).minimize(loss)

nr_epochs = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training
    for e in range(nr_epochs):
        total_loss = 0.0

        for t in range(0, len(cyrData), batch_size):
            feed_dict = {input_images: cyrData[t:t+batch_size],
                         output_images: latData[t:t+batch_size] }

            _, v_loss = sess.run([training_step, loss], feed_dict=feed_dict)
            total_loss += v_loss

        print('Epoch {} - Total loss: {}'.format(e + 1, total_loss))

    # Testing
    n = 8
    figure = plt.figure()
    count = 0
    for i in range(n):
        feed_dict = {input_images: cyrData[i*3000:i*3000+1],
                     output_images: latData[i*3000:i*3000+1]}
        # Encode and decode the digit image
        g = sess.run(tconv2, feed_dict=feed_dict)
        A = g[0,:,:,0]
        figure.add_subplot(3,8,1+count)
        plt.imshow(cyrData[i*3000,:,:,0])
        plt.axis('off')
        figure.add_subplot(3,8,9+count)
        plt.imshow(latData[i*3000,:,:,0])
        plt.axis('off')
        figure.add_subplot(3,8,17+count)
        plt.imshow(A)
        plt.axis('off')
        count += 1


plt.savefig('sample.png')