import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

def make_img_nice (I):
    min= np.min(I, axis=0)
    max = np.max(I, axis=0)

    norm_type = (min <= 0)
    scale_factor = np.maximum(np.abs(norm_type * min), np.abs(max))
    shift = np.multiply(norm_type, 127)

    I = np.divide(I, scale_factor)
    I = np.multiply(I, 127)
    return np.add(I, shift)

a = 20
bb = 1.2

def sigmoid_modified(x):
    return 1 / (1 + np.exp(-x * a))

'''
Shift images values, such that they have zero mean.
Then apply a slightly modified sigmoid function in order to remove noise
(try to make background white and emphasize letters)
Afterwards flatten the images
'''
def normalize(I):
    x_queer = np.sum(np.sum(I, axis=2, keepdims=True), axis=1, keepdims=True) / (784 * 255)
    train_x = sigmoid_modified((I / 255) - (x_queer/bb))
    # Here you can see the changes the normalization does to the images.
    '''for i in range(0,10):
        plt.subplot(1, 2, 1)
        plt.imshow(I[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(train_x[i], cmap='gray')
        plt.show()'''
    return np.reshape(train_x, (train_x.shape[0], 784))

# Divide each pixel by the l2-norm, in order to get unit-length pixel
def unit_length(x):
    return np.divide(x, np.reshape(np.linalg.norm(x, axis=1, ord=2), [-1,1]))

# Normalize pixel values to [0,1]. Then shift them, such that they have zero mean.
def zero_mean(x):
    return x/255 - np.reshape(np.mean(x/255, axis=1), [-1,1])

'''
load the data.
Normalize in two different ways:
1. train_x_s and test_x_s uses the normalize method(which transforms to zero mean and then applies the sigmoid function)
2. train_x_u and test_x_u uses the unit_length() and zero_mean() function (as described on the assignment sheet.
Training data is split into 43 batches of size 463 images in order to make training faster.
'''
def load_sample_dataset():
    dataset = 'train_test_file_list.h5'
    with h5py.File(dataset, 'r') as hf:
        x_in = np.array(hf.get('train_x'), dtype=np.float64)
        train_x_s = np.split(normalize(x_in), 43, axis=0)
        train_x_u = np.split(unit_length(zero_mean(np.reshape(x_in, [19909, 784]))), 43, axis=0)

        y_in = np.squeeze(np.array(hf.get('train_y')))
        train_y = np.zeros((y_in.shape[0], 10))
        train_y[np.arange(y_in.shape[0]), y_in] = 1
        train_y = np.split(train_y, 43, axis=0)

        x_2 = np.array(hf.get('test_x'), dtype=np.float64)
        test_x_s =normalize(x_2)
        test_x_u = unit_length(zero_mean(np.reshape(x_2, [3514, 784])))

        y_test_in = np.squeeze(np.array(hf.get('test_y')))
        test_y = np.zeros((y_test_in.shape[0], 10))
        test_y[np.arange(y_test_in.shape[0]), y_test_in] = 1

    return train_x_s, train_x_u, train_y, test_x_s, test_x_u, test_y, x_in, y_in


train_x_s, train_x_u, train_y, test_x_s, test_x_u, test_y, x_in, y_in = load_sample_dataset()

# Basic logistic regressor model.
x = tf.placeholder(tf.float64, shape=[None, 784])
y_ = tf.placeholder(tf.float64, shape=[None, 10])

W = tf.Variable(tf.random_normal([784, 10], dtype=tf.float64), dtype=tf.float64)
b = tf.Variable(tf.random_normal([10], dtype=tf.float64), dtype=tf.float64)

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# The GradienDescentOptimizer is used to train the data which was normalized with the sigmoid function.
train_step_gradient = tf.train.GradientDescentOptimizer(5).minimize(cross_entropy)
# The AdamOptimizer is used to train the data normalized as described on the exercise sheet.
train_step_adam = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def run(unit_length, train_step):
    # train the model using the GradienDescentOptimizer and sigmoid normalized data
    if unit_length == False:
        i = 0
        # This training loop seemd to yield good results
        for j in range(5):
            train_step = tf.train.GradientDescentOptimizer(10 / 10 ** (j*2)).minimize(cross_entropy)
            for _ in range(1000):
                batch_xs = train_x_s[i]
                batch_ys = train_y[i]
                i = (i + 1) % 43
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            print('train accuracy: %s, test accuracy: %s' % (sess.run(accuracy, feed_dict={x: train_x_s[i % 43], y_: train_y[i % 43]}), sess.run(accuracy, feed_dict={x: test_x_s, y_: test_y})))

        acc = 0
        for i in range(len(train_x_s)):
            acc += sess.run(accuracy, feed_dict={x: train_x_s[i], y_: train_y[i]})
        acc /= len(train_x_s)
        print('total train accuracy: %s' % acc)
        return sess.run(W)
    # This uses the AdamOptimizer and the normalization recommended on the exercise sheet
    else:
        for i in range(5000):
            sess.run(train_step, feed_dict={x: train_x_u[i%43], y_: train_y[i%43]})
            if i % 1000 == 0:
                print('train accuracy: %s, test accuracy: %s' % (sess.run(accuracy, feed_dict={x: train_x_u[i%43], y_: train_y[i%43]}), sess.run(accuracy, feed_dict={x: test_x_u, y_: test_y})))
        acc = 0
        for i in range(len(train_x_u)):
            acc += sess.run(accuracy, feed_dict={x: train_x_u[i], y_: train_y[i]})
        acc /= len(train_x_u)
        print('total train accuracy: %s' % acc)
        return sess.run(W)


# We get better results with the second method.


print('Classify with unit length normalization and AdamOptimizer:')
W_adam = run(True, train_step_adam)
W_adam = np.reshape(make_img_nice(W_adam), (28,28,10))

print('\n Now with sigmoid normalization and Gradient descent:')
W_grad = run(False, train_step_gradient)
W_grad = np.reshape(make_img_nice(W_grad), (28,28,10))

pics = np.zeros((10,10))
for i in range(10):
    idx = 0
    num_found = 0
    while(num_found < 10):
        if y_in[idx] == i:
            pics[num_found, i] = idx
            num_found += 1
        idx += 1

print(pics)

plt.figure(figsize = (11,10))
gs1 = gridspec.GridSpec(11, 10)
gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes.

for i in range(10):
    ax1 = plt.subplot(gs1[i])
    plt.axis('off')
    plt.imshow(W_grad[:, :, i], cmap='gray')
for i in range(10):
    for j in range(10):
        ax1 = plt.subplot(gs1[10 + 10 * i + j])
        plt.axis('off')
        plt.imshow(x_in[int(pics[i, j]), :, :], cmap='gray')

plt.show()

